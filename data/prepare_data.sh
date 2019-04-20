cat data/raw/train/news-commentary-v9.fr-en.en | perl data/normalize-punctuation.perl > data/cleaned/train/news-commentary-v9.fr-en.en
cat data/raw/train/news-commentary-v9.fr-en.fr | perl data/normalize-punctuation.perl > data/cleaned/train/news-commentary-v9.fr-en.fr
cat data/raw/train/giga-fren.release2.fixed.fr | perl data/normalize-punctuation.perl > data/cleaned/train/giga-fren.release2.fixed.fr
cat data/raw/train/giga-fren.release2.fixed.en  | perl data/normalize-punctuation.perl > data/cleaned/train/giga-fren.release2.fixed.en