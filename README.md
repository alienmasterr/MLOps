# MLOps

| Property                                     | Evidence a reviewer checks                                                               | Added in |
| :------------------------------------------- | :--------------------------------------------------------------------------------------- | :------- |
| _Repository reproducible from a clean clone_ | _CI is green;_ `make setup` _and_ `make lint` _succeed on a fresh clone_                 | _Lab 1_  |
| _Design document exists and is current_      | `docs/design.md` _follows the template, every section filled or marked as an assumption_ | _Lab 1_  |

Online platforms receive far more comments than human moderators can read, and the language of toxic comments keeps changing.
This project builds a system that scores every new comment for toxicity and decides: publish, send to human review, or hide.
It uses the Jigsaw Unintended Bias (Civil Comments) dataset, split by time so that new months can be replayed as new batches.
The system watches its own inputs and predictions for drift, and a human approves any retrained model before it goes live.
The goal is to catch most toxic comments quickly while keeping the human review queue small and the whole system reproducible from this repository.

Full design: docs/design.md
