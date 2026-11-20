# misc (evaluation-01 / task01)

Files here are not necessary to build `../evaluation-01.html`.

## Origin of the marker tool

- Source: `course-hydrodynamics/resources/homeworks/homework02/marks.js` and `marks.css`
  (workspace `workspace-master.course.y2.block02-03-Hydrodynamics`).
- Copies with changes: `../evaluation-01/marks.js`, `../evaluation-01/marks.css`.
- Changes in `marks.js`:
  - file name `marks.json` -> `evaluation-01-markers.json` (all messages and dialogs);
  - localStorage key `hw02-marks-v1` -> `dte-eval01-task01-marks-v1`;
  - IndexedDB name `hw02-marks` -> `dte-eval01-task01-marks`;
  - `BLOCK_SEL`: add `pre`, remove display math (this report has no MathJax).
- Changes in `marks.css`: a dark block at the end (colours from homework02 `theme.css`).
