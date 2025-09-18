from __future__ import annotations
import os
import re
import json
from typing import List, Tuple
from datetime import datetime, timezone, timedelta
import humanize  # pip install humanize
from flask import (
    Flask, request, redirect, url_for, render_template, send_from_directory,
    flash, abort, jsonify
)
from werkzeug.utils import secure_filename
from slugify import slugify
import requests
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-for-manga-viewer")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
METADATA_FILE = os.path.join(BASE_DIR, "manga_metadata.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_SUFFIX = "_typeset.png"  # strict suffix
COVER_ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Load metadata
def load_metadata():
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:  # file kosong
                    return {}
                return json.loads(content)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] metadata.json invalid, reset ke kosong: {e}")
            return {}
    return {}


def save_metadata(metadata):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def get_manga_metadata(code):
    metadata = load_metadata()
    return metadata.get(code, {"genres": [], "description": ""})

def get_manga_metadata(code: str) -> dict:
    """Ambil metadata manga dari manga_metadata.json"""
    try:
        with open("manga_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata.get(code, {})
    except Exception as e:
        print(f"Error load metadata: {e}")
        return {}


from datetime import datetime

def update_manga_metadata(code, genres=None, description=None, chapter=None, pages: int = 0):
    metadata = load_metadata()
    now = datetime.utcnow().isoformat(timespec="seconds")

    if code not in metadata:
        metadata[code] = {
            "genres": [],
            "description": "",
            "created_at": now,
            "updated_at": now,
            "chapters": {}
        }

    # Simpan genre & deskripsi kalau ada
    if genres is not None and genres.strip():
        metadata[code]["genres"] = [g.strip() for g in genres.split(",") if g.strip()]

    if description is not None and description.strip():
        metadata[code]["description"] = description

    # Update info chapter
    if chapter:
        metadata[code].setdefault("chapters", {})
        metadata[code]["chapters"][chapter] = {
            "uploaded_at": now,
            "pages": pages
        }
        metadata[code]["updated_at"] = now

    save_metadata(metadata)
    return metadata[code]


# ----------------------- Utilities -----------------------
def list_mangas() -> List[str]:
    try:
        return sorted([
            d for d in os.listdir(UPLOAD_FOLDER)
            if os.path.isdir(os.path.join(UPLOAD_FOLDER, d))
        ], key=str.lower)
    except FileNotFoundError:
        return []

def chapter_sort_key(ch: str) -> Tuple:
    parts = ch.split(".")
    key = []
    for p in parts:
        if p.isdigit():
            key.append((0, int(p)))
        else:
            nums = re.findall(r"\d+", p)
            if nums:
                key.extend([(0, int(n)) for n in nums])
            else:
                key.append((1, p.lower()))
    return tuple(key)

def list_chapters(code: str) -> List[str]:
    manga_dir = os.path.join(UPLOAD_FOLDER, code)
    if not os.path.isdir(manga_dir):
        return []
    chapters = [
        d for d in os.listdir(manga_dir)
        if os.path.isdir(os.path.join(manga_dir, d))
    ]
    return sorted(chapters, key=chapter_sort_key)

def is_typeset_png(filename: str) -> bool:
    return filename.lower().endswith(ALLOWED_SUFFIX)

def sorted_typeset_pages(chapter_dir: str) -> List[str]:
    if not os.path.isdir(chapter_dir):
        return []
    pages = [f for f in os.listdir(chapter_dir) if is_typeset_png(f)]
    def page_key(name: str):
        m = re.match(r"(\d+).*" + re.escape(ALLOWED_SUFFIX) + r"$", name)
        if m:
            return (0, int(m.group(1)))
        return (1, name.lower())
    return sorted(pages, key=page_key)

def get_cover_filename(code: str) -> str | None:
    base = os.path.join(UPLOAD_FOLDER, code)
    for name in ("cover.png", "cover.jpg", "cover.jpeg", "cover.webp"):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            return name
    return None

def infer_cover_from_first_page(code: str) -> str | None:
    chapters = list_chapters(code)
    if not chapters:
        return None
    first_ch = chapters[0]
    cdir = os.path.join(UPLOAD_FOLDER, code, first_ch)
    pages = sorted_typeset_pages(cdir)
    return f"{first_ch}/{pages[0]}" if pages else None

def get_cover_path(code: str) -> str | None:
    fn = get_cover_filename(code)
    if fn:
        return f"{code}/{fn}"
    derived = infer_cover_from_first_page(code)
    if derived:
        return f"{code}/{derived}"
    return None

def is_cover_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in COVER_ALLOWED_EXTS

def paginate(items: List, page: int, per_page: int):
    total = len(items)
    pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, pages))
    start = (page - 1) * per_page
    end = start + per_page
    meta = {
        "page": page,
        "pages": pages,
        "total": total,
        "has_prev": page > 1,
        "has_next": page < pages,
        "prev_page": page - 1 if page > 1 else None,
        "next_page": page + 1 if page < pages else None,
    }
    return items[start:end], meta

def format_title(code: str) -> str:
    """
    Format slug menjadi judul proper.
    Contoh:
    'nia-liston-the-merciless-maiden-official' 
    -> 'Nia Liston: The Merciless Maiden (Official)'
    """
    # pisahkan kata
    words = code.replace("_", "-").split("-")
    title_parts = []
    suffix_parts = []

    for w in words:
        lw = w.lower()
        if lw in {"official", "eng", "raw", "jp"}:
            suffix_parts.append(w.capitalize())
        elif lw == "the":
            title_parts.append("The")
        else:
            title_parts.append(w.capitalize())

    title = " ".join(title_parts)
    if suffix_parts:
        title += " (" + " ".join(suffix_parts) + ")"

    return title

def get_manga_item(code: str):
    cover_rel = get_cover_path(code)
    chapters = list_chapters(code)
    total_pages = 0
    latest_mtime = 0
    manga_dir = os.path.join(UPLOAD_FOLDER, code)

    metadata = get_manga_metadata(code)

    # cari chapter terbaru dari metadata
    latest_chapter = None
    if "chapters" in metadata and metadata["chapters"]:
        try:
            # normal format dict {"ch1": {"uploaded_at": "...", "pages": 12}, ...}
            latest_chapter = max(
                metadata["chapters"].items(),
                key=lambda kv: kv[1]["uploaded_at"] if isinstance(kv[1], dict) else str(kv[1])
            )[0]
        except Exception:
            # fallback kalau value masih string (legacy format)
            latest_chapter = max(metadata["chapters"].keys(), key=chapter_sort_key)

    for ch in chapters:
        cdir = os.path.join(manga_dir, ch)
        pages = sorted_typeset_pages(cdir)
        total_pages += len(pages)
        mtimes = [os.path.getmtime(os.path.join(cdir, p)) for p in pages] if pages else []
        mtime = max(mtimes) if mtimes else os.path.getmtime(cdir)
        latest_mtime = max(latest_mtime, mtime)

    return {
        "code": code,
        "title": format_title(code),
        "cover_rel": cover_rel,
        "total_pages": total_pages,
        "total_chapters": len(chapters),
        "latest_mtime": latest_mtime,
        "genres": metadata.get("genres", []),
        "description": metadata.get("description", ""),
        "created_at": metadata.get("created_at"),
        "updated_at": metadata.get("updated_at"),
        "chapters_meta": metadata.get("chapters", {}),
        "latest_chapter": latest_chapter
    }

def humanize_time_diff(ts: str | float) -> str:
    """
    Terima timestamp ISO (str) atau epoch (float),
    lalu kembalikan string waktu relatif.
    """
    if not ts:
        return "??"
    try:
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts)
        else:
            dt = datetime.fromtimestamp(ts)
        now = datetime.now(timezone.utc)
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)

        diff = now - dt
        seconds = int(diff.total_seconds())

        if seconds < 60:
            return f"{seconds} sec ago"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes} min ago"
        elif seconds < 86400:
            hours = seconds // 3600
            return f"{hours} hrs ago"
        else:
            days = seconds // 86400
            return f"{days} day{'s' if days > 1 else ''} ago"
    except Exception as e:
        print("timeago error:", e)
        return "??"

@app.template_filter("timeago")
def timeago_filter(value):
    return humanize_time_diff(value)

@app.template_filter("dt")
def fmt_datetime(epoch: float, mode: str = "relative"):
    try:
        dt = datetime.fromtimestamp(epoch)
        if mode == "date":
            return dt.strftime("%d %b %Y")
        else:
            # relative time
            diff = datetime.now() - dt
            seconds = int(diff.total_seconds())
            if seconds < 60:
                return f"{seconds} sec ago"
            elif seconds < 3600:
                minutes = seconds // 60
                return f"{minutes} min ago"
            elif seconds < 86400:
                hours = seconds // 3600
                return f"{hours} hrs ago"
            else:
                days = seconds // 86400
                return f"{days} day{'s' if days > 1 else ''} ago"
    except Exception:
        return "??"
    
@app.template_filter("dt")
def fmt_datetime(epoch: float, mode: str = "absolute"):
    try:
        if not epoch:
            return "-"
        dt = datetime.fromtimestamp(epoch)

        if mode == "relative":
            diff = datetime.now() - dt
            seconds = int(diff.total_seconds())

            if seconds < 60:
                return f"{seconds} sec ago"
            elif seconds < 3600:
                return f"{seconds // 60} min ago"
            elif seconds < 86400:
                return f"{seconds // 3600} hours ago"
            elif seconds < 2592000:
                return f"{seconds // 86400} days ago"
            elif seconds < 31536000:
                return f"{seconds // 2592000} months ago"
            else:
                return f"{seconds // 31536000} years ago"

        return dt.strftime("%d %b %Y")  # absolute
    except Exception:
        return "-"

# ----------------------- Generate Genres -----------------------
@app.route('/api/generate-genres', methods=['POST'])
def generate_genres():
    try:
        data = request.get_json()
        title = data.get('title', '')

        GEMINI_API_KEY = "KEY"

        if GEMINI_API_KEY and GEMINI_API_KEY != 'your-gemini-api-key':
            prompt = f"""
Tugas Anda adalah mengidentifikasi SEMUA genre manga yang paling relevan untuk judul "{title}".
- Jawab HANYA dengan daftar genre.
- Pisahkan dengan koma tanpa angka, bullet, atau teks tambahan.
- Jangan sertakan kata pengantar atau penutup.
Contoh format jawaban:
Action, Adventure, Fantasy, Shounen, Supernatural
"""

            response = requests.post(
                f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}',
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    genres_text = result['candidates'][0]['content']['parts'][0]['text']
                    genres = [g.strip() for g in genres_text.split(',') if g.strip()]
                else:
                    genres = ["Action", "Adventure", "Fantasy"]
            else:
                genres = ["Action", "Adventure", "Fantasy"]
        else:
            genres = ["Action", "Adventure", "Fantasy"]

        return jsonify({'genres': genres})

    except Exception as e:
        print(f"Error generating genres: {e}")
        return jsonify({'genres': ["Action", "Adventure", "Fantasy"]})


# ----------------------- Generate Description -----------------------
@app.route('/api/generate-description', methods=['POST'])
def generate_description():
    try:
        data = request.get_json()
        title = data.get('title', '')

        # Pastikan 'genres' selalu string
        genres_input = data.get('genres', [])
        genres_str = ", ".join(genres_input) if isinstance(genres_input, list) else str(genres_input)

        GEMINI_API_KEY = "AIzaSyDkLBZmzcGJGIOKgVQTUDoA-zH43URt4-8"

        if GEMINI_API_KEY and GEMINI_API_KEY != 'your-gemini-api-key':
            prompt = f"""
Buat sinopsis singkat (2-4 kalimat) untuk manga berjudul "{title}" dengan genre {genres_str}.
- Gunakan bahasa Indonesia.
- Fokus pada premis utama, konflik awal, atau keunikan cerita.
- Buat pembaca penasaran, tanpa menyebut kata 'sinopsis'.
- Jangan beri judul atau pembuka, langsung mulai ceritanya.
"""

            response = requests.post(
                f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}',
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    description = result['candidates'][0]['content']['parts'][0]['text']
                else:
                    description = f"{title} adalah sebuah manga seru yang penuh dengan petualangan dan aksi."
            else:
                description = f"{title} adalah sebuah manga seru yang penuh dengan petualangan dan aksi."
        else:
            description = f"{title} adalah sebuah manga seru yang penuh dengan petualangan dan aksi. Ikuti perjalanan seru para karakter."

        return jsonify({'description': description.strip()})

    except Exception as e:
        print(f"Error generating description: {e}")
        return jsonify({'description': f"{title} adalah manga yang menarik dengan cerita yang seru."})
    
# ----------------------- Routes -----------------------
@app.template_filter("relativedt")
def fmt_relative(epoch_or_iso):
    """
    Format ke relative time (ex: '2 days ago').
    Bisa terima epoch float/int atau ISO string.
    """
    try:
        if isinstance(epoch_or_iso, (int, float)):
            dt = datetime.fromtimestamp(epoch_or_iso)
        elif isinstance(epoch_or_iso, str):
            dt = datetime.fromisoformat(epoch_or_iso)
        else:
            return "-"
        return humanize.naturaltime(datetime.utcnow() - dt)
    except Exception:
        return "-"
    
@app.route("/")
def index():
    items = [get_manga_item(code) for code in list_mangas()]
    items.sort(key=lambda x: x["latest_mtime"], reverse=True)

    page = int(request.args.get("page", 1))
    page_items, meta = paginate(items, page, per_page=12)

    # Inject absolute cover URLs
    for it in page_items:
        it["cover_url"] = url_for("serve_upload", path=it["cover_rel"]) if it["cover_rel"] else None

    return render_template("index.html", items=page_items, meta=meta)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        manga_code = request.form.get("manga_code", "").strip()
        chapter = request.form.get("chapter", "").strip()
        genres = request.form.get("genres", "").strip()
        description = request.form.get("description", "").strip()
        auto_slug = request.form.get("auto_slug") == "on"

        if auto_slug or not manga_code:
            manga_code = slugify(manga_code or request.form.get("manga_title", "")) or "untitled"
        manga_code = secure_filename(manga_code)
        if not manga_code:
            flash("Manga code tidak valid.", "error")
            return redirect(url_for("upload"))

        chapter = re.sub(r"[^0-9A-Za-z._-]", "", chapter)
        files = request.files.getlist("files")
        cover_file = request.files.get("cover")

        if not chapter and not cover_file:
            flash("Isi chapter atau unggah cover terlebih dahulu.", "error")
            return redirect(url_for("upload"))

        manga_dir = os.path.join(UPLOAD_FOLDER, manga_code)
        os.makedirs(manga_dir, exist_ok=True)

        saved, skipped = 0, []
        if chapter:
            target_dir = os.path.join(manga_dir, chapter)
            os.makedirs(target_dir, exist_ok=True)

            for f in files:
                if not f or not f.filename:
                    continue
                filename = secure_filename(f.filename)
                if not is_typeset_png(filename):
                    skipped.append(filename)
                    continue
                dest = os.path.join(target_dir, filename)
                f.save(dest)
                saved += 1

            # update metadata with chapter upload time
            # sebelum ini tadi ada dua kali panggilan
            # ganti jadi satu kali aja di paling akhir
            update_manga_metadata(manga_code, genres if genres else None, description if description else None, chapter if chapter else None)

        # Handle optional cover upload
        if cover_file and cover_file.filename:
            cover_name = secure_filename(cover_file.filename)
            if is_cover_file(cover_name):
                ext = os.path.splitext(cover_name)[1].lower()
                cover_dest = os.path.join(manga_dir, f"cover{ext}")
                cover_file.save(cover_dest)
                flash("Cover berhasil diunggah.", "success")
            else:
                flash("Cover harus berupa PNG/JPG/JPEG/WEBP.", "warning")

        if saved:
            flash(f"Berhasil mengunggah {saved} file ke {manga_code}/{chapter}.", "success")
        if skipped:
            flash(f"Lewati {len(skipped)} file (wajib berakhiran '{ALLOWED_SUFFIX}').", "warning")

        return redirect(url_for("gallery", code=manga_code))

    return render_template("upload.html", mangas=list_mangas())

@app.route("/manga/<code>")
def gallery(code: str):
    code = secure_filename(code)
    chapters = list_chapters(code)
    
    # Get cover URL for this manga
    cover_rel = get_cover_path(code)
    cover_url = url_for("serve_upload", path=cover_rel) if cover_rel else None
    
    # Get metadata
    metadata = get_manga_metadata(code)
    
    if not chapters:
        flash("Manga belum memiliki chapter atau tidak ditemukan.", "info")
    
    return render_template("gallery.html", code=code, chapters=chapters, 
                          cover_url=cover_url, metadata=metadata)

@app.route("/manga/<code>/<chapter>")
def view_chapter(code: str, chapter: str):
    code = secure_filename(code)
    chapter = re.sub(r"[^0-9A-Za-z._-]", "", chapter)
    chapter_dir = os.path.join(UPLOAD_FOLDER, code, chapter)
    if not os.path.isdir(chapter_dir):
        abort(404)
    pages = sorted_typeset_pages(chapter_dir)

    # Prev/Next navigation
    chapters = list_chapters(code)   # ✅ ambil semua chapter
    try:
        idx = chapters.index(chapter)
    except ValueError:
        idx = -1
    prev_ch = chapters[idx - 1] if idx > 0 else None
    next_ch = chapters[idx + 1] if (idx != -1 and idx + 1 < len(chapters)) else None

    return render_template(
        "view.html",
        code=code,
        chapter=chapter,
        pages=pages,
        chapters=chapters,   # ✅ kirim ke template
        prev_ch=prev_ch,
        next_ch=next_ch
    )

@app.route('/uploads/<path:path>')
def serve_upload(path):
    return send_from_directory(UPLOAD_FOLDER, path)

@app.get('/api/chapters/<code>')
def api_chapters(code: str):
    code = secure_filename(code)
    chapters = list_chapters(code)
    next_suggest = None
    numeric = [c for c in chapters if re.fullmatch(r"\d+(?:\.\d+)*", c)]
    if numeric:
        last = sorted(numeric, key=chapter_sort_key)[-1]
        if re.fullmatch(r"\d+", last):
            next_suggest = str(int(last) + 1)
    return {"chapters": chapters, "suggest_next": next_suggest}

# ----------------------- New Routes -----------------------
@app.route('/library')
def library():
    items = [get_manga_item(code) for code in list_mangas()]
    items.sort(key=lambda x: x["code"].lower())
    
    page = int(request.args.get("page", 1))
    page_items, meta = paginate(items, page, per_page=24)
    
    for it in page_items:
        it["cover_url"] = url_for("serve_upload", path=it["cover_rel"]) if it["cover_rel"] else None
    
    return render_template("library.html", items=page_items, meta=meta)

@app.route('/categories')
def categories():
    # Get all genres from metadata
    metadata = load_metadata()
    all_genres = set()
    for manga_data in metadata.values():
        all_genres.update(manga_data.get("genres", []))
    
    # Add some default genres if empty
    if not all_genres:
        all_genres = {"Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror", "Romance", "Sci-Fi"}
    
    # Count manga per genre
    genre_count = {}
    for genre in all_genres:
        count = sum(1 for manga_data in metadata.values() if genre in manga_data.get("genres", []))
        genre_count[genre] = count
    
    return render_template("categories.html", genres=sorted(all_genres), genre_count=genre_count)

@app.route('/category/<genre>')
def category(genre: str):
    # Get all manga with this genre
    metadata = load_metadata()
    manga_with_genre = []
    
    for code, manga_data in metadata.items():
        if genre in manga_data.get("genres", []):
            manga_item = get_manga_item(code)
            manga_with_genre.append(manga_item)
    
    manga_with_genre.sort(key=lambda x: x["latest_mtime"], reverse=True)
    
    page = int(request.args.get("page", 1))
    page_items, meta = paginate(manga_with_genre, page, per_page=24)
    
    for it in page_items:
        it["cover_url"] = url_for("serve_upload", path=it["cover_rel"]) if it["cover_rel"] else None
    
    return render_template("category.html", genre=genre, items=page_items, meta=meta)

@app.route('/latest')
def latest():
    items = [get_manga_item(code) for code in list_mangas()]
    items.sort(key=lambda x: x["latest_mtime"], reverse=True)
    
    page = int(request.args.get("page", 1))
    page_items, meta = paginate(items, page, per_page=24)
    
    for it in page_items:
        it["cover_url"] = url_for("serve_upload", path=it["cover_rel"]) if it["cover_rel"] else None
    
    return render_template("latest.html", items=page_items, meta=meta)

@app.route('/search')
def search():
    query = request.args.get('q', '').lower().strip()
    items = []
    
    if query:
        metadata = load_metadata()
        for code in list_mangas():
            # Search in code/title
            if query in code.lower():
                items.append(get_manga_item(code))
            # Search in genres and description
            elif code in metadata:
                manga_data = metadata[code]
                if any(query in genre.lower() for genre in manga_data.get("genres", [])) or \
                   (manga_data.get("description") and query in manga_data.get("description", "").lower()):
                    items.append(get_manga_item(code))
    
    items.sort(key=lambda x: x["latest_mtime"], reverse=True)
    
    page = int(request.args.get("page", 1))
    page_items, meta = paginate(items, page, per_page=24)
    
    for it in page_items:
        it["cover_url"] = url_for("serve_upload", path=it["cover_rel"]) if it["cover_rel"] else None
    
    return render_template("search.html", items=page_items, meta=meta, query=query)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
