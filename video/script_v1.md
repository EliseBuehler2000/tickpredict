# TickPredict — Congressional App Challenge 2026 video script (v1)

**Target length:** about 2:45 (rules: 1–3 minutes; judges may penalize videos outside that). About 370 spoken words, plus two optional lines.
**Upload:** public on YouTube or Vimeo. **Deadline:** 12:00 pm EDT, Monday, October 26, 2026.

**Rubric checklist** (2026 rulebook, "Demonstration video"): names of everyone on the team ✔ (Intro) · app name ✔ (Intro) · purpose in one clear sentence ✔ (Intro) · target audience ✔ (Intro) · tools and coding languages ✔ (How I built it) · the app working ✔ (Demo).

---

## 1. Opening (0:00–0:20)

[B-roll: a trail or backyard in New Hampshire; someone checking their ankle or a dog after a walk]

- Here in New Hampshire, we have one of the highest rates of Lyme disease in the country.
- Every year, about 476,000 Americans are diagnosed and treated for it.
- It usually starts with a tick you find on yourself, and no way to know if it matters.
- Is it a deer tick? How long was it attached? Should I call a doctor?

## 2. Intro (0:20–0:40)

[Face to camera, then the TickPredict home screen]

- Hi, I'm [NAME(S)], and this is TickPredict.
- **TickPredict identifies a tick from a single photo and tells you what it means for Lyme disease, right on your phone.**
- It's for anyone who spends time outdoors: hikers, parents, farmers, and people in rural towns like mine, where the nearest doctor can be a long drive away.
- It runs entirely on the phone, so it works without internet, and your photos stay private.

## 3. Demo (0:40–1:50)

[Screen recording or phone-over-shoulder shots. Use a real tick if you have one, otherwise a printed photo.]

- Let's say I just found this tick. I open TickPredict and take a photo.
- [Analyzing screen] While it works, it shows each thing it's checking.
- [Species card] It's a blacklegged tick, or deer tick, the species that spreads Lyme disease, and it lists what else this tick can carry.
- [Life stage and engorgement cards] It's a nymph, and it's semi-engorged, so it was probably attached for a while.
- [Lyme risk card] Putting that together, it explains that a doctor may offer a preventive antibiotic if the tick came off in the last 72 hours.
- [Tap Find care nearby] One tap finds urgent care nearby.
- [Save, then History] And I can save the scan, with where the bite was, in case symptoms show up weeks later.
- [Optional, 5 seconds: scan a dog tick or lone star tick] If it's a different species, like this American dog tick, it tells you that instead. Not every tick carries Lyme.

## 4. How I built it (1:50–2:35)

[Visuals: training charts or terminal, the DGX Spark, a grid of tick photos, Xcode]

- I trained two neural networks on more than 13,000 tick photos: research photos from the University of Pennsylvania, and real phone photos from iNaturalist.
- I trained them in Python with PyTorch on an NVIDIA DGX Spark, then converted them with Core ML to run directly on the iPhone.
- On photos the models never saw during training, TickPredict tells deer ticks apart from other ticks 97% of the time.
- The app is written in Swift with SwiftUI, and uses MapKit to find care.
- [Optional, about 8 seconds, if you have time] To make the app see photos exactly the way the models were trained to, I rewrote the image-resizing math from Python in Swift, so it gives the same answer on the phone as on the computer.

## 5. Wrap-up (2:35–2:50)

[Face to camera, then end card with the app name and GitHub link]

- Lyme disease is easiest to prevent in the first few days, but only if you know the tick matters.
- TickPredict gives you that answer in seconds, anywhere, even with no signal.
- Thank you!

---

## Notes for recording

- **Read time:** about 370 words at a natural pace (about 150 a minute) is roughly 2:30, which leaves room for the 5-second analysis animation and taps. Time a full read before filming, and cut the two optional lines first if you run long.
- **Purpose sentence:** the bold line in the Intro is the "one, clear sentence" the rules ask for. Say it slowly.
- **Names:** every team member must be named in the video.
- **AI disclosure:** the 2026 rules require all AI use to be fully disclosed in the submission materials, and AI must not be the whole of the technical development. Decide how you'll disclose it in the written answers, and consider a line in the video's tools section, such as "I also used AI coding assistants for parts of the code."
- **Accuracy numbers:** 97% (deer tick vs. other) and 85% (exact species) come from 1,304 iNaturalist photos held out from training. They were also used to choose the best model, so they may run slightly high. For final numbers, run the untouched test set once (1,577 iNaturalist and 60 Penn photos) before submitting.
- **Sources for the opening:** New Hampshire has "one of the highest incidence rates of Lyme in the country" (NH DHHS). About 476,000 Americans are diagnosed and treated each year (CDC estimate from insurance claims, 2010–2018). The CDC notes this figure may overcount; it is a count of people treated, not confirmed cases.
