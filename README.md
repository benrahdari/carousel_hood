# Under the Hood of Carousels

Code and de-identified interaction data for:

> Behnam Rahdari et al. **Under the Hood of Carousels: Investigating User Engagement and Navigation Effort in
> Multi-list Recommender Systems.** IUI '25. https://doi.org/10.1145/3708359.3712130

The study ran on CARE, a platform for carousel user studies
(https://doi.org/10.1145/3640544.3645223).

## The study in brief
170 crowd workers each completed 10 rounds of movie choice. A round has a 20 s **browsing** phase (look around,
open a movie's details) and a 10 s **decision** phase (click "Select this movie"). If the decision phase runs
out, the round ends with no pick. The page holds about 50 movies from the MovieLens top titles.

**Picks were incentive-compatible:** participants were told to pick the movie they wanted most, and were entered
in a raffle to win a picked movie (about $8 in credit). This is not described in the paper.

Each participant was assigned to one cell of a 2 x 2 x 2 between-subject design, balanced on arrival:

| factor | levels |
|---|---|
| `ui` | `carousel` (genre rows) or `list` |
| `adaptation` (`type` in the app) | `adaptive`: after each pick, movies sharing a genre with it are boosted (+5%) and movies in genres seen but not picked are lowered (-1%); `control`: no updates |
| `baseline` | `popularity` (most-rated MovieLens titles) or `random` (a fixed random sample of 50) initial page |

## Repository layout
```
release/            de-identified data (CSV) + codebook (release/README.md), CC BY 4.0
scripts/            export_events.py: raw study export -> release/ files
study_app/          the Flask application used to run the study
  app.py
  templates/
  data/             MovieLens ml-latest-small + movie metadata (see study_app/data/README.md)
```

## Data
See [`release/README.md`](release/README.md) for every file and column. In short: `sessions.csv` (one row per
participant, condition and survey), `shown_titles.csv` (the movies on each participant's page) and `events.csv`
(every click, scroll, 1 Hz mouse sample and non-character key press, with the movie id and on-page position of
each click). Participants are labelled P000-P169; crowd ids, dates and typed characters are removed.

## Running the study app
```bash
pip install -r requirements.txt
export FLASK_SECRET_KEY=<a long random string>
cd study_app
flask --app app run
```
Open `/start` to be assigned to a condition, or `/study?type=adaptive&baseline=random&ui=carousel` to pick one.
Logs are written to `study_app/experiment_data/` (git-ignored). Two instruction images and the consent form in
`templates/recommendations.html` point to the original study server and need to be replaced for a new deployment.

## Citation
Please cite the IUI '25 paper above. If you use the data, also cite the dataset: https://doi.org/10.5281/zenodo.24361731

## License
Code: MIT (`LICENSE`). Data in `release/`: CC BY 4.0. MovieLens data: MovieLens license.
