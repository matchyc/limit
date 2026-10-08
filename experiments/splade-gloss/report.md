# SPLADE++ on LIMIT-small original and gloss queries

Model `naver/splade-cocondenser-ensembledistil` on cuda. 46 docs, 1000 original queries, 126 gloss queries. 5.4s. Whole documents are 161 tokens median, max 173; none truncated at 512.

Recall is macro over the two golds, in percent.

## Original wording

| score | R@2 | R@10 | both@2 |
| --- | ---: | ---: | ---: |
| bm25 | 99.15 | 100.0 | 98.4 |
| splade-whole | 92.05 | 99.35 | 86.8 |
| splade-chunk-max | 99.75 | 100.0 | 99.5 |

The 126 attributes that have glosses, original query, documents rewritten to the definition:

| score | R@2 | R@10 | both@2 |
| --- | ---: | ---: | ---: |
| bm25-on-original-docs | 96.83 | 100.0 | 94.44 |
| splade-on-original-docs | 85.32 | 99.21 | 78.57 |
| bm25 | 4.37 | 19.84 | 0.79 |
| splade-whole | 12.3 | 45.24 | 3.97 |

## Gloss queries (`Who likes {definition}?`)

| score | R@2 | R@10 | both@2 |
| --- | ---: | ---: | ---: |
| bm25 | 3.57 | 24.21 | 0.0 |
| bm25-gloss-only | 3.57 | 23.41 | 0.0 |
| splade-whole | 8.33 | 44.44 | 0.79 |
| splade-chunk-max | 16.67 | 53.97 | 9.52 |
| splade-gloss-only | 10.71 | 46.83 | 2.38 |

Query expansion contains a content token of the original attribute: 38/126 (30.16%). Gloss text alone: 35.71%. First gold document still has the attribute token: 51.59%.

When the query expansion recovered the attribute, SPLADE whole R@2 14.47, R@10 57.89.
When it did not, SPLADE whole R@2 5.68, R@10 38.64.

## Examples

- **Ham** ← cured pork cut from a thigh. gold ranks [25, 37]. attribute in query expansion: False {}. top terms: [['thigh', 2.15], ['pork', 2.035], ['thighs', 1.773], ['cut', 1.688], ['cured', 1.638], ['cure', 1.472], ['liked', 1.46], ['like', 1.294]]
- **Joshua Trees** ← spiky mojave yuccas with a forked crown. gold ranks [31, 15]. attribute in query expansion: False {}. top terms: [['fork', 1.701], ['crown', 1.506], ['yu', 1.424], ['##cca', 1.377], ['##ja', 1.325], ['##ik', 1.323], ['liked', 1.118], ['like', 1.08]]
- **Havarti** ← a mild creamy cheese with small irregular holes. gold ranks [13, 5]. attribute in query expansion: False {}. top terms: [['irregular', 2.133], ['cheese', 2.088], ['mild', 1.854], ['holes', 1.778], ['small', 1.668], ['hole', 1.616], ['creamy', 1.105], ['liked', 1.069]]
- **Chairs** ← seats with a back for one sitter. gold ranks [8, 1]. attribute in query expansion: False {}. top terms: [['back', 2.554], ['sit', 2.14], ['one', 2.081], ['seat', 1.842], ['sitting', 1.489], ['seats', 1.407], ['##ter', 1.32], ['want', 1.206]]
- **Ancient Egypt** ← the Nile civilization of pharaohs and pyramids. gold ranks [2, 3]. attribute in query expansion: True {'ancient': 0.497, 'egypt': 1.4935}. top terms: [['nile', 1.998], ['pyramid', 1.888], ['liked', 1.862], ['civilization', 1.497], ['egypt', 1.494], ['like', 1.452], ['pharaoh', 1.295], ['likes', 1.222]]
- **Limes** ← small green citrus fruit. gold ranks [32, 28]. attribute in query expansion: False {}. top terms: [['small', 2.596], ['green', 2.38], ['citrus', 2.082], ['fruit', 1.542], ['liked', 1.5], ['like', 1.472], ['greene', 1.453], ['fruits', 1.284]]
- **Disco Music** ← dance songs from the late seventies with a steady beat. gold ranks [32, 5]. attribute in query expansion: True {'music': 0.933}. top terms: [['steady', 2.323], ['seventies', 2.079], ['late', 1.82], ['dance', 1.691], ['beat', 1.672], ['dancing', 1.39], ['songs', 1.383], ['liked', 1.239]]
- **Soy Sauce** ← a dark salty condiment fermented from beans. gold ranks [38, 31]. attribute in query expansion: True {'sauce': 0.3236}. top terms: [['dark', 2.109], ['salty', 1.718], ['beans', 1.689], ['##diment', 1.551], ['bean', 1.462], ['##rm', 1.323], ['liked', 1.252], ['like', 1.171]]
- **Barley** ← a cereal grain malted for beer and bread. gold ranks [41, 37]. attribute in query expansion: False {}. top terms: [['cereal', 2.062], ['mal', 1.877], ['bread', 1.709], ['beer', 1.698], ['grain', 1.641], ['liked', 1.339], ['like', 1.159], ['want', 1.129]]
- **Honey** ← sweet syrup that bees store in comb. gold ranks [17, 19]. attribute in query expansion: True {'honey': 0.3892}. top terms: [['comb', 2.388], ['syrup', 2.066], ['bee', 1.872], ['bees', 1.862], ['sweet', 1.733], ['store', 1.644], ['liked', 1.265], ['sugar', 1.228]]

