"""Inline profile pictures via the Kitty graphics protocol.

Renders each message's author avatar at the chat header line using
Kitty's image transmission + placement commands. Images are written
directly to stdout (bypassing curses) and re-placed on every chat
redraw so they track scroll.

Layout: avatars are 2 columns wide and 2 rows tall, placed at the
leftmost columns of the chat window. The chat formatter prepends 3
spaces to each header / newline so the avatar doesn't cover text.
"""

import base64
import hashlib
import io
import logging
import os
import sys
import threading
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# Pillow is optional - if not present, inline PFPs are disabled.
try:
    from PIL import Image, ImageDraw
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

# Cell-size fallbacks — PfpRenderer overrides at runtime via detect_cell_aspect().
PFP_COLS = 5
PFP_ROWS = 2
EMOJI_COLS = 3
EMOJI_ROWS = 1
ATTACHMENT_MAX_COLS = 40
ATTACHMENT_MAX_ROWS = 18
CELL_ASPECT = 2.8   # cell height / cell width fallback
ATTACHMENT_THUMB_PX = 768
ATTACHMENT_TIMEOUT_S = 4
EMOJI_SIZE_PX = 48
PFP_SIZE_PX = 64
KITTY_ID_BASE = 0x70667000   # "pfp\0" — high offset to avoid notification IDs etc.


def kitty_supported():
    """Heuristic check via env vars — no stdin read-back (curses owns it)."""
    term = os.environ.get("TERM", "")
    if "kitty" in term:
        return True
    if os.environ.get("TERM_PROGRAM") == "kitty":
        return True
    if os.environ.get("TERM_PROGRAM") == "ghostty":
        return True
    if os.environ.get("KITTY_WINDOW_ID"):
        return True
    return False


def _send(payload):
    """Write a Kitty graphics escape sequence to stdout."""
    try:
        os.write(sys.stdout.fileno(), payload)
    except OSError:
        pass


def detect_cell_aspect():
    """Probe the controlling terminal for cell pixel dimensions via
    TIOCGWINSZ. Returns cell_h / cell_w as a float, or None if the
    terminal didn't report pixel dimensions (most non-Kitty TTYs).
    Kitty (and Ghostty) fill ws_xpixel/ws_ypixel so this works without
    a stdin-blocking CSI query.
    """
    try:
        import fcntl
        import termios
        for path in ("/dev/tty", None):
            try:
                fd = os.open(path, os.O_RDONLY) if path else sys.stdout.fileno()
                buf = fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\0" * 8)
                if path:
                    os.close(fd)
                import struct
                rows, cols, xpix, ypix = struct.unpack("HHHH", buf)
                if rows > 0 and cols > 0 and xpix > 0 and ypix > 0:
                    cw = xpix / cols
                    ch = ypix / rows
                    return ch / cw
            except OSError:
                continue
    except ImportError:
        pass
    return None


def best_square(rows, cell_aspect):
    """Return (cols, rows) closest to a 1:1 visual square at the given
    cell aspect. rows is fixed (callers choose layout height); cols is
    picked from {floor, ceil} of rows×cell_aspect, whichever lands
    closer to a true square."""
    ideal_c = rows * cell_aspect
    best = None
    for c in (max(1, int(ideal_c)), max(1, int(ideal_c) + 1)):
        visual = c / (rows * cell_aspect)
        err = abs(visual - 1.0) if visual >= 1.0 else abs(1.0 / visual - 1.0)
        if best is None or err < best[0]:
            best = (err, c)
    return best[1], rows


def _apply_circular_mask(im):
    """Return an RGBA copy of `im` with a circular alpha mask applied.
    Pixels outside the inscribed circle become transparent so the
    Kitty renderer composites the chat bg through the corners — i.e.
    Discord-style round avatars.

    Anti-aliases via a 4x super-sampled mask so the edge doesn't look
    jagged at our small placement sizes (we transmit at PFP_SIZE_PX
    and Kitty stretches to ~5 cells wide).
    """
    if not HAVE_PIL:
        return im
    w, h = im.size
    if w <= 0 or h <= 0:
        return im
    scale = 4
    mask = Image.new("L", (w * scale, h * scale), 0)
    ImageDraw.Draw(mask).ellipse(
        (0, 0, w * scale - 1, h * scale - 1), fill=255,
    )
    mask = mask.resize((w, h), Image.LANCZOS)
    out = im.copy()
    if "A" in out.getbands():
        # Intersect existing alpha with circle — preserves transparent holes.
        from PIL import ImageChops
        out.putalpha(ImageChops.multiply(out.getchannel("A"), mask))
    else:
        out.putalpha(mask)
    return out


def _chunk_payload(data, controls):
    """Build Kitty APC sequence(s), chunked at 4096 base64 chars with m=1."""
    b64 = base64.standard_b64encode(data)
    chunks = []
    chunk_size = 4096
    pos = 0
    first = True
    while pos < len(b64):
        chunk = b64[pos:pos + chunk_size]
        pos += chunk_size
        more = pos < len(b64)
        if first:
            ctrl = controls + (",m=1" if more else "")
        else:
            ctrl = ("m=1" if more else "m=0")
        chunks.append(b"\x1b_G" + ctrl.encode("ascii") + b";" + chunk + b"\x1b\\")
        first = False
    return b"".join(chunks)


class PfpRenderer:
    """Owns the avatar cache and pushes images into the terminal."""

    def __init__(self, cache_path, discord, enabled=True):
        self.discord = discord
        self.cache_path = cache_path
        self.enabled = enabled and HAVE_PIL and kitty_supported()
        self._transmitted = {}   # avatar_id -> kitty image id
        self._fetching = set()
        self._next_id = KITTY_ID_BASE
        self._placed = set()
        self._next_placement_id = 1
        self._lock = threading.Lock()
        detected = detect_cell_aspect()
        self.cell_aspect = detected or CELL_ASPECT
        self.pfp_cols, self.pfp_rows = best_square(2, self.cell_aspect)
        self.emoji_cols, self.emoji_rows = best_square(1, self.cell_aspect)
        self._attach_px = {}   # url -> (sw, sh) for source crops
        logger.info(
            f"pfp cell_aspect={self.cell_aspect:.3f} "
            f"(detected={detected is not None}) "
            f"pfp={self.pfp_cols}x{self.pfp_rows} "
            f"emoji={self.emoji_cols}x{self.emoji_rows}"
        )

    def _alloc_id(self, avatar_id):
        with self._lock:
            if avatar_id in self._transmitted:
                return self._transmitted[avatar_id]
            kid = self._next_id
            self._next_id += 1
            self._transmitted[avatar_id] = kid
            return kid

    def _avatar_path(self, user_id, avatar_id):
        """Return path to cached 64px PNG. Downloads+converts on miss."""
        # `pfp_round_` prefix busts pre-round cache.
        png_name = f"pfp_round_{avatar_id}.png"
        png_path = os.path.join(os.path.expanduser(self.cache_path), png_name)
        if os.path.isfile(png_path):
            return png_path
        webp_path = self.discord.get_pfp(
            user_id, avatar_id, size=PFP_SIZE_PX, save_path=self.cache_path,
        )
        if not webp_path or not os.path.isfile(webp_path):
            return None
        try:
            with Image.open(webp_path) as im:
                im = im.convert("RGBA").resize(
                    (PFP_SIZE_PX, PFP_SIZE_PX), Image.LANCZOS,
                )
                im = _apply_circular_mask(im)
                im.save(png_path, format="PNG")
        except Exception as e:
            logger.debug(f"pfp convert failed for {avatar_id}: {e}")
            return None
        return png_path

    def _ensure_transmitted(self, user_id, avatar_id):
        """If the image isn't on the terminal yet, transmit it.

        Returns the Kitty image id, or None if not yet ready.
        """
        if avatar_id in self._transmitted:
            return self._transmitted[avatar_id]
        if avatar_id in self._fetching:
            return None
        self._fetching.add(avatar_id)
        try:
            path = self._avatar_path(user_id, avatar_id)
            if not path:
                return None
            kid = self._alloc_id(avatar_id)
            with open(path, "rb") as f:
                data = f.read()
            ctrl = f"a=t,f=100,i={kid},q=2"
            _send(_chunk_payload(data, ctrl))
            return kid
        finally:
            self._fetching.discard(avatar_id)

    def clear_placements(self):
        """Remove all current avatar placements from the terminal.

        Uses lowercase d=i which deletes placements but preserves the
        image data in Kitty's storage, so we don't have to re-transmit
        the bytes on the next placement.
        """
        if not self.enabled or not self._placed:
            return
        for kid in self._placed:
            _send(b"\x1b_Ga=d,d=i,i=" + str(kid).encode("ascii") + b",q=2\x1b\\")
        self._placed.clear()


    def invalidate_transmissions(self):
        """Forget which images have been sent to the terminal.

        Call this after anything that drops Kitty's image storage (e.g.
        the full-screen clear we issue on tree toggle). Forces the next
        place() to re-transmit the image bytes.
        """
        with self._lock:
            self._transmitted.clear()
            self._placed.clear()
            self._next_id = KITTY_ID_BASE

    def _emoji_path(self, emoji_id):
        """Download + convert a custom emoji to a square PNG. Returns
        local path or None on failure.
        """
        png_name = f"emoji_{emoji_id}.png"
        png_path = os.path.join(os.path.expanduser(self.cache_path), png_name)
        if os.path.isfile(png_path):
            return png_path
        webp_path = self.discord.get_emoji(emoji_id, size=EMOJI_SIZE_PX)
        if not webp_path or not os.path.isfile(webp_path):
            return None
        try:
            with Image.open(webp_path) as im:
                im = im.convert("RGBA").resize(
                    (EMOJI_SIZE_PX, EMOJI_SIZE_PX), Image.LANCZOS,
                )
                im.save(png_path, format="PNG")
        except Exception as e:
            logger.debug(f"emoji convert failed for {emoji_id}: {e}")
            return None
        return png_path

    def _ensure_emoji_transmitted(self, emoji_id):
        """Transmit a custom-emoji image to Kitty if not already there.

        Cache key is the emoji_id (a numeric string) — distinct from
        avatar hashes so the two namespaces don't collide.
        """
        key = f"emoji:{emoji_id}"
        if key in self._transmitted:
            return self._transmitted[key]
        if key in self._fetching:
            return None
        self._fetching.add(key)
        try:
            path = self._emoji_path(emoji_id)
            if not path:
                return None
            with self._lock:
                kid = self._next_id
                self._next_id += 1
                self._transmitted[key] = kid
            with open(path, "rb") as f:
                data = f.read()
            ctrl = f"a=t,f=100,i={kid},q=2"
            _send(_chunk_payload(data, ctrl))
            return kid
        finally:
            self._fetching.discard(key)

    def _attachment_path(self, url):
        """Download and PNG-thumbnail an arbitrary image URL.

        Cached by URL hash; returns local path or None on failure.
        Synchronous — caller is the draw thread, so this blocks. The
        cache-hit path is fast; first load of each image incurs the
        download cost once.
        """
        if not HAVE_PIL:
            return None
        key = hashlib.sha256(url.encode("utf-8", "replace")).hexdigest()[:16]
        png_path = os.path.join(os.path.expanduser(self.cache_path), f"attach_{key}.png")
        if os.path.isfile(png_path):
            return png_path
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "endcord"})
            with urllib.request.urlopen(req, timeout=ATTACHMENT_TIMEOUT_S) as resp:
                raw = resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            logger.debug(f"attachment fetch failed {url}: {e}")
            return None
        try:
            with Image.open(io.BytesIO(raw)) as im:
                im = im.convert("RGB")
                im.thumbnail((ATTACHMENT_THUMB_PX, ATTACHMENT_THUMB_PX), Image.LANCZOS)
                im.save(png_path, format="PNG")
        except Exception as e:
            logger.debug(f"attachment convert failed {url}: {e}")
            return None
        return png_path

    def measure_attachment(
        self,
        url,
        max_cols=ATTACHMENT_MAX_COLS,
        max_rows=ATTACHMENT_MAX_ROWS,
        cell_aspect=None,
    ):
        """Download the image if needed and return (cols, rows) for an
        aspect-preserving placement.

        Returns None if the image can't be loaded. The returned cell
        dimensions fit within (max_cols, max_rows) and approximate the
        source pixel aspect ratio given the terminal's cell_aspect
        (cell_height / cell_width).
        """
        if not HAVE_PIL:
            return None
        if cell_aspect is None:
            cell_aspect = self.cell_aspect
        path = self._attachment_path(url)
        if not path:
            return None
        try:
            with Image.open(path) as im:
                sw, sh = im.size
        except Exception as e:
            logger.debug(f"attachment measure failed {url}: {e}")
            return None
        if sw <= 0 or sh <= 0:
            return None
        # c/r = (sw/sh) * cell_aspect for undistorted display.
        ratio = (sw / sh) * cell_aspect
        if ratio >= max_cols / max_rows:
            cols = max_cols
            rows = max(1, round(cols / ratio))
        else:
            rows = max_rows
            cols = max(1, round(rows * ratio))
        # Stash source-pixel dims for partial-render crops below.
        self._attach_px[url] = (sw, sh)
        return cols, rows

    def _ensure_attachment_transmitted(self, url):
        """Transmit attachment PNG to Kitty if not already; return image id."""
        key = f"attach:{url}"
        if key in self._transmitted:
            return self._transmitted[key]
        if key in self._fetching:
            return None
        self._fetching.add(key)
        try:
            path = self._attachment_path(url)
            if not path:
                return None
            with self._lock:
                kid = self._next_id
                self._next_id += 1
                self._transmitted[key] = kid
            with open(path, "rb") as f:
                data = f.read()
            ctrl = f"a=t,f=100,i={kid},q=2"
            _send(_chunk_payload(data, ctrl))
            return kid
        finally:
            self._fetching.discard(key)

    def place_attachment(self, url, row, col, cols=ATTACHMENT_MAX_COLS, rows=ATTACHMENT_MAX_ROWS,
                         crop_top_cells=0, full_rows=None):
        """Place attachment thumbnail at (row, col). crop_top_cells>0 = render bottom slice only."""
        if not self.enabled or not url:
            return
        kid = self._ensure_attachment_transmitted(url)
        if kid is None:
            return
        with self._lock:
            pid = self._next_placement_id
            self._next_placement_id += 1
        cup = f"\x1b[{row + 1};{col + 1}H".encode("ascii")
        if crop_top_cells > 0 and full_rows and url in self._attach_px:
            sw, sh = self._attach_px[url]
            y_off = max(0, min(sh - 1, int(crop_top_cells * sh / full_rows)))
            h_src = max(1, sh - y_off)
            controls = (
                f"a=p,i={kid},p={pid},c={cols},r={rows},Y={y_off},H={h_src},"
                f"W={sw},X=0,C=1,q=2"
            )
        else:
            controls = f"a=p,i={kid},p={pid},c={cols},r={rows},C=1,q=2"
        place = (f"\x1b_G{controls}\x1b\\").encode("ascii")
        _send(cup + place)
        self._placed.add(kid)

    def place_emoji(self, emoji_id, row, col, cols=None, rows=None):
        """Place a custom emoji at terminal cell (row, col)."""
        if not self.enabled or not emoji_id:
            return
        if cols is None:
            cols = self.emoji_cols
        if rows is None:
            rows = self.emoji_rows
        kid = self._ensure_emoji_transmitted(emoji_id)
        if kid is None:
            return
        with self._lock:
            pid = self._next_placement_id
            self._next_placement_id += 1
        cup = f"\x1b[{row + 1};{col + 1}H".encode("ascii")
        place = (
            f"\x1b_Ga=p,i={kid},p={pid},c={cols},r={rows},C=1,q=2\x1b\\"
        ).encode("ascii")
        _send(cup + place)
        self._placed.add(kid)

    def place(self, user_id, avatar_id, row, col):
        """Place this user's avatar at terminal cell (row, col).

        Uses Kitty's a=p with C=1 (don't move cursor) so we can write
        the placement without disturbing curses' notion of cursor pos.
        Position is set via the standard CUP escape before the place.

        Each call gets a fresh placement_id so multiple messages from
        the same author render side-by-side instead of the later one
        moving the earlier placement.
        """
        if not self.enabled or not avatar_id:
            return
        kid = self._ensure_transmitted(user_id, avatar_id)
        if kid is None:
            return
        with self._lock:
            pid = self._next_placement_id
            self._next_placement_id += 1
        cup = f"\x1b[{row + 1};{col + 1}H".encode("ascii")
        place = (
            f"\x1b_Ga=p,i={kid},p={pid},c={self.pfp_cols},r={self.pfp_rows},C=1,q=2\x1b\\"
        ).encode("ascii")
        _send(cup + place)
        self._placed.add(kid)
