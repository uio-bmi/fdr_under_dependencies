import scrapy

from web_scraping.utils import parse_article


class ClinicalEpigeneticsSpider(scrapy.Spider):
    name = 'clinical_epigenetics'
    start_urls = ['https://clinicalepigeneticsjournal.biomedcentral.com/articles?query=&searchType=&tab=keyword']

    BASE_URL = 'https://clinicalepigeneticsjournal.biomedcentral.com'

    def parse(self, response):
        for article in response.css('h3.c-listing__title a'):
            link = article.css('::attr(href)').get()
            absolute_url = self.BASE_URL + link
            yield scrapy.Request(absolute_url, callback=parse_article)

        next_page = response.css('a[rel="next"]::attr(href)').get()
        if next_page:
            absolute_next_page = response.urljoin(next_page)
            self.logger.info(f"Following next page: {absolute_next_page}")
            yield scrapy.Request(absolute_next_page, callback=self.parse)
