import scrapy
from web_scraping.utils import parse_article


class NARSpider(scrapy.Spider):
    name = 'nar'
    allowed_domains = ['academic.oup.com']
    start_urls = ['https://academic.oup.com/nar/issue']

    def parse(self, response):
        year_links = response.css('select#YearsList option::attr(value)').getall()
        for year_link in year_links:
            yield response.follow(year_link, callback=self.parse_year)

        next_page = response.css('a[rel="next"]::attr(href)').get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)

    def parse_year(self, response):
        issue_links = response.css('select#IssuesList option::attr(value)').getall()
        for issue_link in issue_links:
            yield response.follow(issue_link, callback=self.parse_issue)

    def parse_issue(self, response):
        article_links = response.css('h5.customLink.item-title a::attr(href)').getall()
        for link in article_links:
            full_url = response.urljoin(link)
            yield scrapy.Request(full_url, callback=self.parse_and_print)

        next_page = response.css('a[rel="next"]::attr(href)').get()
        if next_page:
            yield response.follow(next_page, callback=self.parse_issue)

    def parse_and_print(self, response):
        for item in parse_article(response):
            self.logger.info(f"YIELDING ITEM: {item}")
            yield item
