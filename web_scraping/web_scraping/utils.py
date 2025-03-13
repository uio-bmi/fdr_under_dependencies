import re


def parse_article(response):
    title = response.css('h1.c-article-title::text').get()
    abstract = " ".join(response.css('section.c-abstract p::text').getall())

    full_text_sections = response.css('div.c-article-section')
    filtered_text = []

    for section in full_text_sections:
        section_title = section.css('h2::text, h3::text').get(default="").lower()
        if "references" not in section_title and "bibliography" not in section_title:  # Skip reference sections
            filtered_text.append(" ".join(section.css('p::text').getall()))

    full_text = " ".join(filtered_text)

    publication_year = response.css('time::attr(datetime)').get()
    if publication_year:
        publication_year = int(re.search(r'\d{4}', publication_year).group())
    else:
        publication_year = 0

    if publication_year < 2020:
        return

    article_title_abstract = title + " " + abstract
    article_title_abstract_lower = article_title_abstract.lower()
    article_text = abstract + " " + full_text
    article_text_lower = article_text.lower()

    methylation_keywords = ["methylation", "epigenetic", "epigenomic"]

    stat_test_keywords = ["statistical test", "t-test", "t test", "anova", "wilcoxon", "kruskal-wallis",
                          "kruskal wallis", "mann–whitney u",  "mann whitney u", "mann–whitney–wilcoxon",
                          "mann whitney wilcoxon", "wilcoxon mann whitney", "wilcoxon-mann-whitney", "rank-sum",
                          "rank sum", "linear model", "linear regression", "limma", "edgeR", "deseq2",
                          "hypothesis test", "p-value", "p value", "f-test", "f test"]

    multiple_correction_keywords = ["multiple testing", "multiple comparison", "multiple correction",
                                    "multiple adjustment", "multiple hypothesis", "adjusted p-value",
                                    "adjusted p value", "family-wise error rate", "family wise error rate",
                                    "fwer", "false discovery rate", "fdr", "q value", "q-value"]

    bh_keywords = ["benjamini-hochberg", "benjamini hochberg", "bh"]
    bonferroni_keywords = ["bonferroni"]
    by_keywords = ["benjamini-yekutieli", "benjamini yekutieli"]
    by_keywords_case_sensitive = ["BY"]

    mentions_methylation = keyword_match(article_title_abstract_lower, methylation_keywords)
    if mentions_methylation:
        mentions_stat_test = keyword_match(article_text_lower, stat_test_keywords)
        mentions_multiple_correction = keyword_match(article_text_lower, multiple_correction_keywords)
        yield {
            'title': title,
            'link': response.url,
            'publication_year': publication_year,
            'mentions_methylation': mentions_methylation,
            'mentions_stat_test': mentions_stat_test,
            'mentions_multiple_correction': mentions_multiple_correction,
            'mentions_bh': keyword_match(article_text_lower, bh_keywords),
            'mentions_bonferroni': keyword_match(article_text_lower, bonferroni_keywords),
            'mentions_by': (
                keyword_match(article_text_lower, by_keywords) or
                keyword_match(article_text, by_keywords_case_sensitive)
            )
        }


def keyword_match(text, keywords):
    return any(re.search(rf'\b{re.escape(kw)}s?\b', text) for kw in keywords)
