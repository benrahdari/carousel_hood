# Released interaction logs

Interaction data from the study reported in:

> Under the Hood of Carousels: Investigating User Engagement and Navigation Effort in Multi-list Recommender Systems. IUI '25. https://doi.org/10.1145/3708359.3712130

The study ran on CARE (https://doi.org/10.1145/3640544.3645223). 170 crowd participants browsed one page of
about 50 movies, shown either as carousels or as a single list, over several timed rounds, and picked a movie
in each round. **Picks were incentive-compatible:** participants were told to pick movies they actually wanted,
and could win a picked title in a later draw. (This incentive is not described in the paper.)

Between-subject conditions (2 x 2 x 2):
- `ui`: carousel | list
- `adaptation`: adaptive (page adapts to the participant's actions) | control
- `baseline`: popularity | random (how the initial titles were chosen and ordered)

## De-identification
- Participants are labelled P000-P169. Crowd-platform and study ids are not included.
- Times are seconds from each participant's first logged event; calendar dates and clock times are removed.
- Keyboard events keep only non-character keys (Control, Tab, ArrowDown, ...); any letter, digit or symbol is replaced by `char`.
- No free-text answers were collected.

## Files

### sessions.csv: one row per participant (170)
| column | meaning |
|---|---|
| participant | P000-P169 |
| ui, adaptation, baseline | condition (see above) |
| mouse_logging_ms | mouse sampling interval (1000 ms) |
| n_events | logged events |
| session_secs | first to last logged event, seconds |
| age_band | 1 = no answer, 2 = 18-24, 3 = 25-34, 4 = 35-44, 5 = 45-54, 6 = 55+ |
| digital_comfort | 1 = no answer, 2 = not comfortable ... 6 = very comfortable |
| internet_usage | 1 = no answer, 2 = rarely ... 6 = very often |

Survey items are sliders that start at 1, so 1 can also mean the slider was left untouched. Blank = no survey submitted (27 participants).

### shown_titles.csv: one row per participant x title on the page (8,014)
| column | meaning |
|---|---|
| participant | P000-P169 |
| group | carousel row name, or `list` |
| rank | position within the row or list (0-based) |
| movieId | MovieLens movie id |
| title, year, genres | from MovieLens |

This is the page as stored at the end of the session. The same titles stay on the page across rounds; in the
adaptive condition their order can change between rounds, and only the final order is stored.

### events.csv: one row per logged event (117,974)
| column | meaning |
|---|---|
| participant | P000-P169 |
| t | seconds since the participant's first logged event |
| event | mousemove (sampled every 1 s) \| scroll \| click \| keydown |
| x, y | pointer position in the viewport, px |
| scroll_x, scroll_y | scroll offsets (scroll events) |
| movieId | movie under the click, blank if none |
| row_position, col_position, row_genre | position and row of the clicked tile |
| on_movie | whether the click landed on a movie tile |
| target_id, target_class | DOM id and class of the click target |
| key | key name for keydown events |

How to read clicks:
- **Pick:** `target_id` = `selectthismovie`, or `target_class` contains `btn-select`.
- **Open a movie's details:** a click with a `movieId` whose `target_class` contains one of `Askpopover`, `carousel_image`, `movie-item`, `movie-title`, `list_image`, `mt-3`, `text-center`.
- **Study flow:** `target_id` values `DynamicModalButton`, `dynamicModal`, `start`, `currenttask`, `phase-name`, `time-remaining` belong to instructions and round dialogs.

## Known limitations
- The mouse is sampled once per second.
- Which tiles were in the viewport is not logged; it can be inferred only from layout and scroll offsets.
- Rounds are not marked explicitly in the log; picks are the most reliable round boundary.
- Participants made 4.4 picks on average, and 18 made none. A round in which the 10 s decision phase ran out without a pick advances to the next round and logs no selection (see `study_app/app.py`, phase `end_of_decision`), so missing picks are timeouts, not lost data.

## Code
`scripts/export_events.py` builds the three files from the raw study export (not included). The study application is in `study_app/`.

## License
Data in this folder: CC BY 4.0 (see `LICENSE`).
