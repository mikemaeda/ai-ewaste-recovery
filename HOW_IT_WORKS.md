# How the e-waste detector works — explained simply

This is a plain-English walkthrough you can read out loud to explain the whole
project to someone who knows nothing about AI. No jargon. Analogies included.

---

## 1. The big idea (the 30-second version)

> "We built an AI that looks at a live video or a photo, draws a box around each
> piece of electronic waste it sees — a circuit board, a cable, a battery — tells
> you what it is, and then tells you which valuable materials are inside it
> (like copper, gold, or lithium) and roughly how much they're worth."

Why it matters: e-waste is full of valuable metals. If a machine can instantly
recognise what's in a pile of junk, recyclers know what's worth recovering.

---

## 2. The difference between "classification" and "detection"

This is the one concept to get across:

- **Classification** (the project's older `train.py` model) looks at a whole
  photo and says *one* word: "this is a battery." It can't tell you *where* the
  battery is, and it assumes there's only one thing in the picture.

- **Detection** (the new YOLO model) is smarter. It can find *several* objects
  in the same image and draw a **box** around each one: "battery here, cable
  there, circuit board in the corner."

> Analogy: classification is like asking "what's this photo of?" Detection is
> like asking "point to everything in this photo and name each one."

We use detection because real e-waste comes in messy piles, not one neat item
at a time.

---

## 3. What "YOLO" is

YOLO stands for "You Only Look Once." It's a famous, fast object-detection AI.
"You only look once" means it scans the entire image in a single pass instead of
checking thousands of little sub-regions one by one — that's why it's fast
enough to run **live** on a webcam.

We use **YOLO11**, a recent version. We didn't build it from scratch — we take a
version that already learned to see general shapes and edges from millions of
images, and we *teach it our specific objects* on top of that. That trick is
called **transfer learning**.

> Analogy: instead of teaching a baby to see from birth, we hire someone who
> already has great eyesight and just teach them the six e-waste objects we care
> about. Much faster, needs far fewer example photos.

---

## 4. The six things it learns to recognise

`PCB` (circuit board), `cable`, `battery`, `hard_drive`, `metal_component`,
`plastic_housing`.

---

## 5. The pipeline — the 4 big steps

Think of it like teaching, then testing.

### Step A — Label the photos (teaching material)
A computer doesn't magically know what a battery looks like. We have to show it
**examples** and, on each example photo, **draw a box and type the name**. We use
a free tool called **Label Studio** to do this by hand. We already have thousands
of e-waste photos in the `data/images/` folder to draw boxes on.

> Analogy: making flashcards. Each flashcard is a photo with the answer written
> on it.

### Step B — Organise the data (`prepare_yolo_data.py`)
YOLO is picky about *how* the photos and their box-labels are arranged in
folders, and it wants every image the same size (640×640 pixels). This script
does that tidying automatically and splits the photos into two piles:
- **train** (≈80%) — the flashcards it studies from.
- **validation** (≈20%) — flashcards we hide, then use to test if it really
  learned or just memorised.

### Step C — Train the model (`train_yolo_colab.ipynb` on Google Colab)
Training = showing the AI the flashcards over and over until it gets good. This
needs a powerful graphics chip (GPU), which most laptops don't have — so we use
**Google Colab**, a free website that lends you a GPU in your browser.

The result of training is a single file called **`best.pt`**. That file *is* the
trained brain. We download it and drop it into `artifacts/models/best.pt`.

> Analogy: Colab is a free gym with equipment we don't own at home. We go there,
> do the heavy training, and bring home one thing: the trained muscle (`best.pt`).

A note on settings: because we have *few* photos per object, we tell training to
heavily **augment** them — randomly flip, rotate, zoom, and recolour each photo
so one photo acts like many. This stops the AI from just memorising and helps it
handle real-world lighting.

### Step D — Run live detection (`detect_ewaste.py`)
This is the payoff. It opens a window, turns on your webcam (or loads a photo),
runs every frame through the trained `best.pt`, draws the boxes and labels, and
prints the material info to the terminal. Press **Q** to quit.

---

## 6. The "fallback mode" (why it works before we've trained anything)

Training takes time. So we added a safety net: if the `best.pt` file isn't there
yet, the script loads a **generic pre-trained YOLO** that already knows 80
everyday objects (phone, laptop, keyboard, etc.) and *pretends* they're e-waste
by mapping them onto our categories — e.g. it treats a "cell phone" as a
stand-in for a circuit board.

This lets you **see the window working immediately** as a demo. It's not
accurate e-waste detection — it's a placeholder until your own `best.pt` is
ready. We also filter this mode so it **ignores irrelevant things like people**
and only shows objects that map to e-waste.

> Analogy: a demo car with a cardboard engine — it shows you the dashboard lights
> and steering work, while the real engine (`best.pt`) is still being built.

---

## 7. The material info (`material_values.json`)

This is just a lookup table we wrote by hand from published research. For each of
the six objects it stores:
- the **materials inside** and roughly **what percentage** each is (e.g. a circuit
  board is ~20% copper, with tiny amounts of gold and palladium),
- a **recovery priority** (high / medium / low),
- an **estimated value** in dollars per kilogram.

When the detector spots an object, it looks the object up in this table and
prints the top 3 materials. The AI doesn't "know" chemistry — we just connect its
visual answer to our table.

---

## 8. The files, in one line each

| File | What it is |
|------|-----------|
| `detect_ewaste.py` | The live window. Run this to see detection. |
| `prepare_yolo_data.py` | Tidies photos + labels into YOLO's required folders. |
| `train_yolo_colab.ipynb` | The notebook we run on Google Colab to train. |
| `material_values.json` | The hand-written table of materials + values. |
| `requirements_yolo.txt` | The list of software libraries needed. |
| `setup_yolo.ps1` | Installs everything with one command. |
| `best.pt` (after training) | The trained AI brain. Lives in `artifacts/models/`. |
| `yolo11n.pt` | The generic fallback brain (auto-downloaded). |

---

## 9. How to run it (the two commands that matter)

Install once:
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_yolo.ps1
```

See the live window:
```powershell
python detect_ewaste.py --source auto --backend auto --width 1280 --height 720 --conf 0.25
```
(Window opens as a separate window — check your taskbar. Press **Q** to quit.)

If you only want to test the Arducam camera feed without AI:
```powershell
python arducam_viewer.py --source auto --backend auto --width 1280 --height 720
```

Shortcut scripts are included too:
```powershell
powershell -ExecutionPolicy Bypass -File .\run_arducam_viewer.ps1
powershell -ExecutionPolicy Bypass -File .\run_ewaste_dashboard.ps1
```

---

## 10. If someone asks a hard question

- **"Is this real-time?"** Yes — YOLO is fast enough to process every webcam frame.
- **"Did you build the AI from scratch?"** No, and we shouldn't — we used transfer
  learning on top of a proven YOLO11 model. That's standard practice and needs
  far less data.
- **"How accurate is it?"** As accurate as the labeled photos we trained it on.
  More good, varied, hand-labeled examples = better accuracy.
- **"Why does it ignore people / chairs?"** We deliberately filter the model down
  to e-waste-relevant objects so it doesn't get distracted.
- **"What's the hardest part?"** Honestly, the labeling — drawing accurate boxes on
  enough photos by hand. The AI is the easy part once the data is good.
  
```
