# SupportOps AI interface

## Purpose
Operate mode: evaluate a fictional support request, inspect the classifier and read the evidence. The interface presents a single input and its result, not a marketing landing page.

## Visual system
Manrope variable font is self-hosted under SIL OFL. White work surfaces sit on a cool pale green ground. Dark forest green communicates the primary action; amber marks review. Neutral borders separate the input, output and report. The desktop workspace has two columns; mobile becomes a vertical workflow. No decorative imagery or animation is required for the task.

## Tokens
Foreground #172f28, secondary #53675e, primary #164b3d, paper #f4f7f5, surface #ffffff, line #d4dfd8, quiet #e7f0e9. Inputs and buttons have visible focus. Metrics use tabular numerals. Sections and text reflow without horizontal scrolling at the reviewed small viewport.

## State and provenance
Loading, error, empty, rejected classification and generation requiring review are distinct. Values come from the committed training report. Browser classification uses the shipped weights. Semantic recovery and native tool calling belong to the local backend; the public site labels its BM25 and WebLLM paths.
