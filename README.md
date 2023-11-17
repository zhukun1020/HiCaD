# HiCAD

The code and data for our EMNLP-findings 2023 paper [Hierarchical Catalogue Generation for Literature Review: A Benchmark ](https://arxiv.org/abs/2304.03512)  is provided at this repo. If you have any questions, please reach me at [kzhu@ir.hit.edu.cn](kzhu@ir.hit.edu.cn).

## Gathering Data

Our survey papers' metadata are obtained from Kaggle up to the end of April 2023. https://www.kaggle.com/datasets/Cornell-University/arxiv.

0. collect arxiv_metadata  https://www.kaggle.com/datasets/Cornell-University/arxiv.

1. Filter out survey papers from the arxiv papers.

We keep papers containing the words ``survey`` and ``review`` in the title and remove ones with ``book review`` and ``comments``.  We finally select 11,435 papers that are considered to be review papers. 

```shell
python 1_filter_survey.py  --arxivmeta_path='path_to_arxiv-metadata-oai-snapshot.json' --survey_info_path='Path_to_save_survey_info'
```

2. Get the html form of survey papers.

It is straightforward to use a crawler to get all 11,435 papers in PDF format according to the ``arxiv-id``. However, extracting catalogue from PDF files is difficult where structural information is usually dropped during converting.
Therefore, we attempt https://ar5iv.org to get the papers in HTML format.
This website processes articles from arXiv as responsive HTML web pages by converting from LaTeX via ``LaTeXML`` https://github.com/brucemiller/LaTeXML.
Some authors do not upload their LaTeX code, we have to skip these and collect 8,397 papers.

```shell
python 2_craw_from_ar5iv.py --survey_info_path='Path_to_save_survey_info' --survey_html_path='Path_to_save_survey_html'
```

3. Process the html file to get the catalogues.

We obtain the original catalogues by cleaning up the HTML files. 

```shell
python 3_extra_outline_html.py --survey_html_path='Path_to_save_survey_html' --survey_outline_path='Path_to_save_survey_outline'
```

4. Get references for survey papers (You need an API key from https://www.semanticscholar.org/product/api )

We collate the list of reference papers and only keep the valid papers where titles and abstracts exist.

```shell
python 4_craw_ref_list.py  --api_key='Your key'  --survey_outline_path='Path_to_save_survey_outline'  --ref_info_path='Path_to_save_refs_info'
```

5. Integrate inputs and outputs and perform dataset slicing into training set, test set, valid set.

We convert all words to lowercase for subsequent generation and evaluation. Finally, after removing data with less than $5$ catalogue items and less than $10$ valid references, we obtain 7,637 references-catalogue pairs. We replace the serial number from the heading with the level mark <$s_i$> using regex.
We choose the computer science field with the largest number of papers for the experiment and split the 4,507 papers into training (80\%), validation (10\%), and test (10\%) sets.

```shell
python 5_combine_data.py  --survey_info_path='Path_to_save_survey_info'  --survey_outline_path='Path_to_save_survey_outline'  --ref_info_path='Path_to_save_refs_info'  --combine_save_path='Path_to_save_processed_data'  --split_save_path='Path_to_save_processed_data' 
```



## Metrics

```shell
python metrics.py  --prediction_file='Path_to_prediction_file'  --ref_file='Path_to_save_reference_file'  --output_dir='Path_to_save_evaluation'
```

