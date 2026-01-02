import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import datetime
import os

class SubstackScraper:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        }

    def get_soup(self, url):
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return BeautifulSoup(response.content, 'html.parser')
            else:
                print(f"Failed to retrieve {url}: Status {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def scrape_newsletter_archive(self, newsletter_url, limit=20):
        """Scrapes the archive page of a newsletter for article links."""
        archive_url = f"{newsletter_url.rstrip('/')}/archive"
        print(f"Scraping archive: {archive_url}")
        soup = self.get_soup(archive_url)
        if not soup:
            return []

        links = []
        # Substack archive structure typically has posts in div.post-preview
        # Identifying links: usually a.post-preview-title
        
        # Try finding post preview titles
        posts = soup.find_all('div', class_='post-preview')
        
        count = 0
        for post in posts:
            if count >= limit:
                break
                
            title_link = post.find('a', class_='post-preview-title')
            if title_link:
                href = title_link.get('href')
                if href:
                    links.append(href)
                    count += 1
        
        # Fallback if class names differ or updated (Checking simple 'a' tags in main list)
        if not links:
             # Look for standard a tags that look like posts
             all_links = soup.find_all('a')
             for a in all_links:
                 if count >= limit:
                     break
                 href = a.get('href')
                 if href and '/p/' in href and 'comments' not in href:
                     if href not in links:
                         links.append(href)
                         count += 1
                         
        return links

    def scrape_article(self, url):
        """Scrapes content from a single article."""
        print(f"Processing: {url}")
        soup = self.get_soup(url)
        if not soup:
            return None

        article_data = {'url': url}
        
        # Title
        title_tag = soup.find('h1', class_='post-title')
        article_data['title'] = title_tag.get_text(strip=True) if title_tag else "No Title"

        # Subtitle
        subtitle_tag = soup.find('h3', class_='subtitle')
        article_data['subtitle'] = subtitle_tag.get_text(strip=True) if subtitle_tag else ""
        
        # Date
        date_tag = soup.find('div', class_='post-date') # legacy
        if not date_tag:
             # Try other common date classes or meta tags
             meta_date = soup.find('meta', property='article:published_time')
             if meta_date:
                 article_data['date'] = meta_date.get('content')
             else:
                 article_data['date'] = datetime.datetime.now().isoformat()
        else:
            article_data['date'] = date_tag.get_text(strip=True)

        # Author
        author_tag = soup.find('div', class_='profile-name') # This can vary
        if not author_tag:
             author_meta = soup.find('meta', attrs={'name': 'author'})
             article_data['author'] = author_meta.get('content') if author_meta else "Unknown"
        else:
             article_data['author'] = author_tag.get_text(strip=True)

        # Content - This is tricky as structure varies. Usually in div.available-content or just div.body
        content_div = soup.find('div', class_='available-content')
        if not content_div:
            content_div = soup.find('div', class_='markup') # Another common class for body
        
        if content_div:
            # Get text, paragraphs
            article_data['content'] = content_div.get_text(separator='\n\n', strip=True)
        else:
            article_data['content'] = ""

        return article_data

    def run(self, newsletters, limit_per_newsletter=10):
        all_articles = []
        
        for nl in newsletters:
            links = self.scrape_newsletter_archive(nl, limit=limit_per_newsletter)
            print(f"Found {len(links)} articles for {nl}")
            
            for link in links:
                # Some links might be relative
                if not link.startswith('http'):
                    link = f"{nl.rstrip('/')}{link}" 
                
                article = self.scrape_article(link)
                if article and article['content']:
                    # Simple heuristic: ignore very short content (likely paywalled or preview only)
                    if len(article['content']) > 1000: # at least 1000 chars
                        article['newsletter'] = nl
                        all_articles.append(article)
                    else:
                        print(f"Skipping {link} - content too short (possibly paywalled)")
                
                time.sleep(1) # Be polite
        
        df = pd.DataFrame(all_articles)
        output_file = os.path.join(self.output_dir, "articles.csv")
        df.to_csv(output_file, index=False)
        print(f"Scraped {len(df)} articles. Saved to {output_file}")
        return df

if __name__ == "__main__":
    targets = [
        "https://www.lennysnewsletter.com",
        "https://newsletter.pragmaticengineer.com",
        "https://blog.bytebytego.com",
        "https://newsletter.systemdesign.one",
        "https://www.refactoring.fm"
    ]
    
    scraper = SubstackScraper()
    scraper.run(targets, limit_per_newsletter=20)
