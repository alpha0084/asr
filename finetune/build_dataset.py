"""Build the Malayalam-audio -> English-reference eval dataset.

Source: the hand-corrected call set (8 calls, English CA/CX transcripts).
 - "Sreethu" calls (3): per-call folders, one .docx each, clean CA:/CX: turns.
 - "Nexora" calls (5): one combined .docx with inline `Call N - M:SS` markers.
Audio is 8 kHz mono telephony .mpeg; paired to each transcript by duration.

Output: finetune/data/dataset.json = [{id, group, audio, duration_s, ref_en, turns:[{speaker,text}]}].
The English reference (`ref_en`) is the full translated text (speaker labels stripped);
`turns` keeps CA/CX for speaker-role eval. Audio stays in place (paths are absolute).
"""
import json, os, re, subprocess, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Folder holding "Recordings and transcription/" + "Call Transcriptions/".
# Override on the box: set ASR_FT_SRC to wherever you copied the data.
SRC = Path(os.environ.get("ASR_FT_SRC", "/Users/admin/Downloads"))
FLAT = SRC / "Recordings and transcription"
FOLDERS = SRC / "Call Transcriptions"
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
SPK_RE = re.compile(r"^\s*(CA|CX|Agent|Customer|Client)\b\s*(?:\([^)]*\))?\s*[:\-]\s*", re.I)
AGENT = {"ca", "agent"}


def paras(docx: Path) -> list[str]:
    xml = zipfile.ZipFile(docx).read("word/document.xml").decode("utf-8")
    out = []
    for p in ET.fromstring(xml).iter(NS + "p"):
        t = "".join(x.text for x in p.iter(NS + "t") if x.text).strip()
        if t:
            out.append(t)
    return out


def duration_s(audio: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nk=1:nw=1", str(audio)], capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def parse_turns(lines: list[str]) -> list[dict]:
    """Lines -> [{speaker: agent|customer, text}]. A line without a speaker tag
    continues the previous turn."""
    turns = []
    for ln in lines:
        m = SPK_RE.match(ln)
        if m:
            spk = "agent" if m.group(1).lower() in AGENT else "customer"
            turns.append({"speaker": spk, "text": ln[m.end():].strip()})
        elif turns:
            turns[-1]["text"] += " " + ln.strip()
    return [t for t in turns if t["text"]]


def ref_text(turns: list[dict]) -> str:
    return " ".join(t["text"] for t in turns).strip()


def audio_by_duration(target_s: float, pool: list[Path], tol=12.0):
    best, bd = None, 1e9
    for a in pool:
        d = abs(audio_cache[a] - target_s)
        if d < bd:
            best, bd = a, d
    return best if bd <= tol else None


# ---- gather audio + durations ----
all_audio = list(FLAT.glob("*.mpeg")) + list(FOLDERS.glob("Call */*.mpeg"))
audio_cache = {a: duration_s(a) for a in all_audio}

dataset = []

# ---- Sreethu (3): explicit per-call folders ----
for folder in sorted(FOLDERS.glob("Call *")):
    docx = next(folder.glob("*.docx"), None)
    rec = next(folder.glob("*.mpeg"), None)
    if not docx or not rec:
        continue
    ps = paras(docx)
    body = [p for p in ps if not re.match(r"^(CA & CX|Call\s*\d+\s*$)", p, re.I)]
    turns = parse_turns(body)
    dataset.append({
        "id": f"sreethu_{folder.name.split()[-1]}",
        "group": "sreethu", "audio": str(rec), "duration_s": round(audio_cache[rec], 1),
        "ref_en": ref_text(turns), "turns": turns,
    })

# ---- Nexora (5): split the combined docx on inline `Call N - M:SS` markers ----
nex = next(FLAT.glob("Call Transcription 1-2-3-4-5*.docx"), None)
if nex:
    full = "\n".join(paras(nex))
    # Split on the inline `Call N - M:SS` markers (no \b — they run into the prior word
    # like "MarketsCall 1"; require the mm:ss so we don't split on "call" inside a sentence).
    parts = re.split(r"(?i)call\s*(\d+)\s*[–\-]\s*(\d+:\d+)", full)
    # parts = [preamble, num, mmss, body, num, mmss, body, ...]
    nex_pool = [a for a in FLAT.glob("*.mpeg")]
    for i in range(1, len(parts), 3):
        num = parts[i]; mmss = parts[i + 1]; body = parts[i + 2] if i + 2 < len(parts) else ""
        body = re.sub(r"(?i)^\s*mins?\b\s*", "", body)   # 'Mins' leaks from the marker into the body
        lines = [ln.strip() for ln in re.split(r"[\n]", body) if ln.strip()]
        turns = parse_turns(lines)
        target = None
        if mmss:
            m, s = mmss.split(":"); target = int(m) * 60 + int(s)
        rec = audio_by_duration(target, nex_pool) if target else None
        dataset.append({
            "id": f"nexora_{num}", "group": "nexora",
            "audio": str(rec) if rec else None,
            "duration_s": round(audio_cache[rec], 1) if rec else None,
            "marker": mmss, "ref_en": ref_text(turns), "turns": turns,
        })

(OUT / "dataset.json").write_text(json.dumps(dataset, indent=2, ensure_ascii=False))

# ---- report ----
print(f"built {len(dataset)} entries -> {OUT/'dataset.json'}\n")
print(f"{'id':12} {'group':8} {'audio_dur':>9} {'marker':>7} {'turns':>5} {'ref_chars':>9}  audio")
for e in dataset:
    a = Path(e["audio"]).name if e["audio"] else "** NO AUDIO MATCH **"
    print(f"{e['id']:12} {e['group']:8} {str(e.get('duration_s')):>9} {str(e.get('marker') or ''):>7} "
          f"{len(e['turns']):>5} {len(e['ref_en']):>9}  {a}")
miss = [e['id'] for e in dataset if not e['audio']]
if miss:
    print("\n!! unpaired:", miss)
