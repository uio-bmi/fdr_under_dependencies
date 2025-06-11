import scrapy

from web_scraping.utils import parse_article


class NatureCommunicationsSpider(scrapy.Spider):
    name = 'nature_communications'

    BASE_URL = "https://www.nature.com/ncomms/articles?type=article&year="

    def start_requests(self):
        for year in range(2020, 2026):
            url = f"{self.BASE_URL}{year}"
            yield scrapy.Request(url, callback=self.parse, meta={"playwright": True})

    def parse(self, response):
        for article in response.css('article h3 a::attr(href)').getall():
            absolute_url = response.urljoin(article)
            yield scrapy.Request(absolute_url, callback=parse_article, meta={"playwright": True})

        next_page = response.css('li[data-page="next"] a::attr(href)').get()
        if next_page:
            absolute_next_page = response.urljoin(next_page)
            self.logger.info(f"Following next page: {absolute_next_page}")
            yield response.follow(next_page, callback=self.parse, meta={"playwright": True})
