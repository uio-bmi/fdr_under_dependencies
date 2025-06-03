import re
from urllib.parse import urlparse
from dateutil import parser


def parse_article(response):
    parsed_url = urlparse(response.url)
    domain = parsed_url.netloc

    if 'academic.oup.com' in domain:
        yield from parse_nar_article(response)
    elif 'genome.cshlp.org' in domain:
        yield from parse_genome_research_article(response)
    else:
        yield from parse_generic_article(response)


def parse_nar_article(response):
    title = response.css('h1.wi-article-title::text').get()
    if title:
        title = title.strip()
    abstract = " ".join(response.css('section.abstract p::text').getall())

    article_container = response.css('div.widget-items[data-widgetname="ArticleFulltext"]')
    sections = article_container.css('h2.section-title, h3.section-title, p.chapter-para')

    filtered_text = []
    current_section = ""

    for el in sections:
        if el.root.tag in ['h2', 'h3']:
            current_section = el.css('::text').get(default="").lower()
        elif el.root.tag == 'p' and "references" not in current_section and "bibliography" not in current_section:
            filtered_text.append(el.css('::text').get(default="").strip())

    full_text = " ".join(filtered_text)

    publication_date = response.css('meta[name="citation_publication_date"]::attr(content)').get()

    yield from process_article_logic(response.url, title, abstract, full_text, publication_date)


def parse_genome_research_article(response):
    title = " ".join(response.css('h1#article-title-1 *::text').getall()).strip()

    abstract = " ".join(response.css('div.section.abstract p::text').getall()).strip()

    content_selectors = response.css('div.section')
    filtered_text = []

    for section in content_selectors:
        heading = section.css('h2.section-title::text, h3.section-title::text').get(default="").lower()
        if "references" not in heading and "bibliography" not in heading:
            paras = section.css('p::text').getall()
            filtered_text.append(" ".join([p.strip() for p in paras]))

    full_text = " ".join(filtered_text)
    publication_date = response.css('meta[name="DC.Date"]::attr(content)').get()

    yield from process_article_logic(response.url, title, abstract, full_text, publication_date)


def parse_generic_article(response):
    title = response.css('h1.c-article-title::text').get()
    abstract = " ".join(response.css('section.c-abstract p::text').getall())

    full_text_sections = response.css('div.c-article-section')
    filtered_text = []

    for section in full_text_sections:
        section_title = section.css('h2::text, h3::text').get(default="").lower()
        if "references" not in section_title and "bibliography" not in section_title:
            filtered_text.append(" ".join(section.css('p::text').getall()))

    full_text = " ".join(filtered_text)

    publication_date = response.css('time::attr(datetime)').get()

    yield from process_article_logic(response.url, title, abstract, full_text, publication_date)


def process_article_logic(url, title, abstract, full_text, publication_date):
    try:
        parsed_date = parser.parse(publication_date)
    except Exception:
        return

    if parsed_date < parser.parse("2020-01-01") or parsed_date > parser.parse("2025-03-31"):
        return

    publication_year = parsed_date.year

    article_title_abstract = title + " " + abstract
    article_title_abstract_lower = article_title_abstract.lower()
    article_text = abstract + " " + full_text
    article_text_lower = article_text.lower()

    methylation_keywords = ["methylation", "epigenetic", "epigenomic"]
    stat_test_keywords = ["statistical test", "t-test", "t test", "anova", "wilcoxon", "kruskal-wallis",
                          "kruskal wallis", "mann–whitney u", "mann whitney u", "mann–whitney–wilcoxon",
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
        yield {
            'title': title,
            'link': url,
            'publication_year': publication_year,
            'mentions_methylation': mentions_methylation,
            'mentions_stat_test': keyword_match(article_text_lower, stat_test_keywords),
            'mentions_multiple_correction': keyword_match(article_text_lower, multiple_correction_keywords),
            'mentions_bh': keyword_match(article_text_lower, bh_keywords),
            'mentions_bonferroni': keyword_match(article_text_lower, bonferroni_keywords),
            'mentions_by': (
                keyword_match(article_text_lower, by_keywords) or
                keyword_match(article_text, by_keywords_case_sensitive)
            )
        }


def keyword_match(text, keywords):
    return any(re.search(rf'\b{re.escape(kw)}s?\b', text) for kw in keywords)
