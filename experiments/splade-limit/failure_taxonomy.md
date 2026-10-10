# SPLADE failure taxonomy on LIMIT-50k

Model `naver/splade-cocondenser-ensembledistil`. 106.8s. A query fails if SPLADE does not put both golds in the top 2.

Failed queries: 767 / 1000.

## Top-1 false-positive label on failed queries

| label | n | % of failed |
| --- | ---: | ---: |
| name_split | 302 | 39.4 |
| attr_phrase | 3 | 0.4 |
| attr_word | 91 | 11.9 |
| attr_affix | 49 | 6.4 |
| expansion | 214 | 27.9 |
| gold | 108 | 14.1 |

Labels: `name_split` = query tokens collide with the person name (Arta→art, Goebel→go, Snowden→snow). `attr_phrase` = a different likes-item contains the full query as words (Honey Bees). `attr_word` = a query word appears as a whole word in another likes-item (Holly Trees for Birch Trees). `attr_affix` = query word is a prefix/suffix/stem of a likes word (Seahorses, Historical Fiction). `expansion` = none of those; MLM neighbors.

## Ablation: drop colliding documents, keep golds

| filter | R@2 | R@10 | R@100 | both@2 |
| --- | ---: | ---: | ---: | ---: |
| unfiltered | 33.8 | 58.65 | 83.55 | 23.3 |
| drop-name-split | 46.35 | 66.2 | 83.75 | 35.7 |
| drop-attr-phrase | 33.85 | 58.85 | 83.8 | 23.3 |
| drop-attr-word | 40.9 | 67.25 | 89.75 | 29.8 |
| drop-attr-affix | 37.95 | 64.2 | 87.45 | 26.6 |
| drop-name-and-phrase | 46.5 | 66.4 | 84.0 | 35.8 |
| drop-all-string-collisions | 62.5 | 79.6 | 92.65 | 53.0 |

Name-split documents are 0.06% of the corpus but 27.6% of top-10 false positives on failed queries.

String collisions together (name + likes phrase/word/affix) are 445/767 = 58% of failed top-1s. Dropping every such document only lifts R@2 from 33.8 to 62.5. BM25 on the same split is 92.65. The rest is MLM expansion: the query vector fires neighbors that many distractors also fire, while the gold attribute is a weak max-pool hit in a 45-item list.

`gold` at top-1 (108 failed queries) means one gold is rank 1 and the other is not in the top 2.

Some `expansion` labels are still name/query WordPiece with a shorter stem than the detector used (prefix length 4). Examples: `Saxman` vs `saxophones` share `sax`; `Arta` vs `arts` share `art`; query `Eggplants` tokenizes as `egg` + `##pl` + `##ants` and ranks `Eggs Benedict`. Those are the same family as Arta. Clean expansion remains: `Psychology` → `Neuroscience` / `Mental Clarity`, `Philosophy` → documents with no shared token, `Butchers` → `Slaughterbeck` via `slaughter`.

## Examples

### name_split

- `Who likes Birch Trees?` top1 `Reece Birchenough` golds [8, 3] reason {'name': 'name_wp:birch|reece', 'phrase': '', 'word': '', 'affix': ''}
- `Who likes Holland Lops?` top1 `Holland Capuchino` golds [14, 13] reason {'name': 'name_prefix:holland|holland', 'phrase': '', 'word': '', 'affix': ''}
- `Who likes The Sound of Wind in the Trees?` top1 `Windell Mwangi` golds [135, 103] reason {'name': 'name_prefix:wind|windell', 'phrase': '', 'word': 'sound|The Sound of Birdsong', 'affix': ''}
- `Who likes Urban Exploration?` top1 `Yehuda Urbaniak` golds [11, 4] reason {'name': 'name_wp:urban|yehuda', 'phrase': '', 'word': '', 'affix': ''}

### attr_phrase

- `Who likes Apple Cider?` top1 `Cathleen Obringer` golds [2474, 691] reason {'name': '', 'phrase': 'Apple Cider Vinegar', 'word': 'apple|Apple Cider Vinegar', 'affix': ''}
- `Who likes Honey?` top1 `Charley Decesaris` golds [3617, 1856] reason {'name': '', 'phrase': 'Honey Bees', 'word': 'honey|Honey Bees', 'affix': ''}
- `Who likes Fishing?` top1 `Vincenza Otarola` golds [2235, 170] reason {'name': '', 'phrase': 'Magnet Fishing', 'word': 'fishing|Magnet Fishing', 'affix': ''}

### attr_word

- `Who likes Stand-up Comedy?` top1 `Judi Wion` golds [98, 43] reason {'name': '', 'phrase': '', 'word': 'comedy|Comedy Movies', 'affix': ''}
- `Who likes Environmental Engineering?` top1 `Tyrik Skluzacek` golds [508, 5969] reason {'name': '', 'phrase': '', 'word': 'engineering|Biomedical Engineering', 'affix': ''}
- `Who likes Hot Chocolate?` top1 `Einar Ree` golds [36, 8] reason {'name': '', 'phrase': '', 'word': 'chocolate|Dark Chocolate', 'affix': ''}
- `Who likes Apple Juice?` top1 `Maximillian Lennington` golds [737, 2443] reason {'name': '', 'phrase': '', 'word': 'juice|Orange Juice', 'affix': 'apple|apples|Apples to Apples'}

### attr_affix

- `Who likes Blueberries?` top1 `Unique Atteberry` golds [424, 724] reason {'name': '', 'phrase': '', 'word': '', 'affix': 'stem:blueberries|blue|Blue-ringed Octopus'}
- `Who likes Rainbows?` top1 `Celie Scherbert` golds [4, 3] reason {'name': '', 'phrase': '', 'word': '', 'affix': 'stem:rainbows|rainbow|Rainbow Eucalyptus Trees'}
- `Who likes Fire Extinguishers?` top1 `Juluis Elezovic` golds [8, 7] reason {'name': '', 'phrase': '', 'word': '', 'affix': 'fire|fireplaces|Fireplaces'}
- `Who likes Photorealism?` top1 `Isreal Ciaravino` golds [27, 23] reason {'name': '', 'phrase': '', 'word': '', 'affix': 'stem:photorealism|photography|Food Photography'}

### expansion

- `Who likes Philosophy?` top1 `Sharleen Villanueua` golds [4679, 802] reason {'name': '', 'phrase': '', 'word': '', 'affix': ''}
- `Who likes Eggplants?` top1 `Dimitri Duplaga` golds [4, 6] reason {'name': '', 'phrase': '', 'word': '', 'affix': ''}
- `Who likes Cribbage?` top1 `Mathilde Cridlebaugh` golds [2, 3] reason {'name': '', 'phrase': '', 'word': '', 'affix': ''}
- `Who likes Butchers?` top1 `Craig Slaughterbeck` golds [4, 3] reason {'name': '', 'phrase': '', 'word': '', 'affix': ''}

