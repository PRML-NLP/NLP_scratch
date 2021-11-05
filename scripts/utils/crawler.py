import requests
from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta
import json

class NatePannRankingCrawler(object):
    def __init__(self):
        self.date_link = "https://pann.nate.com/talk/ranking/d?stdt="
        self.realtime_ranking = [ "https://pann.nate.com/talk/ranking?rankingType=total",
                                "https://pann.nate.com/talk/ranking?rankingType=life",
                                "https://pann.nate.com/talk/ranking?rankingType=teens"]

    def get_contents(self, days=None, current_time=True, to_json=False):
        contents = []
        print('get links...')
        titles = []
        links = []
        if current_time:
            for link in self.realtime_ranking:
                info = self.nate_pann_ranking(link, type=0)
                if len(info['title'])==len(info['link']):
                    titles.extend(info['title'])
                    links.extend(info['link'])
                
                print('\rData %d are collected...' % len(titles), end="")

        if days is not None:
            d = datetime.today()
            for day in range(days):
                date = d.strftime('%Y%m%d')
                info = self.nate_pann_ranking(self.date_link+date, type=1)
                
                if len(info['title'])==len(info['link']):
                    titles.extend(info['title'])
                    links.extend(info['link'])

                d = d - timedelta(days=1)
                print('\rData %d are collected...' % len(titles), end="")
        
        print()

        # repeatation removement
        contents = {}
        dataframe = []
        for idx in range(len(links)):
            contents[links[idx]] = titles[idx]

        print('get contents...')
        i = 1
        for link in contents.keys():
            print('\r%d / %d' % (i, len(contents.keys())), end='')
            try:
                content = self.get_nate_pann_content(link)
                dataframe.append({'title': contents[link], 'content':content})
                i += 1
            except Exception as e:
                print('Nothing')

        if to_json:
            d = datetime.today()
            today = d.strftime('%Y%m%d')
            d = d - timedelta(days)
            tgt_day = d.strftime('%Y%m%d')
            with open('nate_pann_ranking'+tgt_day+'_'+today+'.json', 'w') as js:
                json.dump(dataframe, js, ensure_ascii=False)

            return True
        else:
            return dataframe

    def nate_pann_ranking(self, page_link, type=0):
        titles = []
        links = []
        for i in range(1, 3):
            res = requests.get(page_link + "&page=" + str(i))
            res.raise_for_status()
            res.encoding = None
            html = res.text
            soup = BeautifulSoup(html, 'html.parser')
            # print(soup)
            ranking_list = soup.find('ul', {'class': 'post_wrap'}).findAll('dt')

            for dt in ranking_list:
                link = dt.find('a')
                real_link = 'https://pann.nate.com' + link.get('href')
                try:
                    titles.append(link.text)
                except:
                    titles.append("")
                links.append(real_link)
                
        return {'title': titles, 'link': links}
    
    def get_nate_pann_content(self, link):
        res = requests.get(link)
        res.raise_for_status()
        res.encoding = None
        html = res.text
        
        soup = BeautifulSoup(html, 'html.parser')
        contentArea = soup.find('div', {'id': 'contentArea'})
        parags = contentArea.findAll('p')
        if len(parags) == 0:
            return contentArea.text.strip()
        
        content = ""
        for parag in parags:
            content += parag.text.replace('&nbsp;', '') + '\n'
        if content == '\n':
            return contentArea.text.strip()
        
        return content.strip()

    

if __name__ == '__main__':
    crawler = NatePannRankingCrawler()
    print(crawler.get_contents(days=300, current_time=True, to_json=True))