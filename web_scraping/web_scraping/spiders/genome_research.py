import scrapy
from web_scraping.utils import parse_article

class GenomeResearchSpider(scrapy.Spider):
    name = 'genome_research'

    def start_requests(self):
        base_url = "https://genome.cshlp.org/content/by/year"
        for year in range(2020, 2025):
            yield scrapy.Request(f"{base_url}/{year}", callback=self.parse_issue_page)
        yield scrapy.Request(f"{base_url}/2025", callback=self.parse_early_2025)

    def parse_issue_page(self, response):
        issue_links = response.css('td.proxy-archive-by-year-month a::attr(href)').getall()
        for link in issue_links:
            full_url = response.urljoin(link)
            yield scrapy.Request(full_url, callback=self.parse_issue_articles)

    def parse_early_2025(self, response):
        for td in response.css('td.proxy-archive-by-year-month'):
            month = td.css('h3::text').get('').lower()
            if month in ['january', 'february', 'march']:
                link = td.css('a::attr(href)').get()
                if link:
                    full_url = response.urljoin(link)
                    yield scrapy.Request(full_url, callback=self.parse_issue_articles)

    def parse_issue_articles(self, response):
        articles = response.css('div.cit-extra a[rel="full-text"]::attr(href)').getall()
        if not articles:
            self.logger.warning(f"No articles found on: {response.url}")
        for link in articles:
            full_url = response.urljoin(link)
            yield scrapy.Request(full_url, callback=parse_article)
