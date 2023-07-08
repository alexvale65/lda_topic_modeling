import warnings

warnings.filterwarnings("ignore")

import traceback
import pandas as pd
import time
from pathlib import Path
import os

data_folder = Path(os.path.dirname(__file__))


import logging


logfile = data_folder / 'logs' / 'crawler_google_scholar' /f"{pd.to_datetime('today').strftime('%b %d %Y %I:%M%p')}.log"
logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    filemode="w",
                    level=logging.INFO,
                    filename=logfile)
logger = logging.getLogger()


from selenium import webdriver
from selenium.webdriver.common.by import By


def runQuery(webdriver):
    logger.info('running query search')
    keywords = '''"data science platform" OR "data science framework" OR "data science model"'''
    searchBar = webdriver.find_element_by_id("gs_hdr_tsi")
    searchBar.send_keys(keywords)
    searchBar.submit()
    time.sleep(15)
    txtResults = webdriver.find_elements(By.CLASS_NAME, "gs_ab_mdw")[1].text
    return txtResults.split(' ')[1]


def selectPages():
    url_base = "https://scholar.google.com.br/" \
               "scholar?start=0&q=%22data+science+platform%22+OR" \
               "+%22data+science+framework%22+OR" \
               "+%22data+science+model%22&hl=pt-BR&as_sdt=0,5"
    pages = [url_base]
    num_pages = 1
    while num_pages <= 97:
        url = "https://scholar.google.com.br/" \
                   f"scholar?start={num_pages}0&q=%22data+science+platform%22+OR" \
                   "+%22data+science+framework%22+OR" \
                   "+%22data+science+model%22&hl=pt-BR&as_sdt=0,5"

        pages.append(url)
        num_pages += 1
    return pages


def infoCollection(webdriver, url_page, num_page):
    try:
        webdriver.get(url_page)
        collectionCiteFormats = []
        linksPaperNavigator = webdriver.find_elements(By.CLASS_NAME, 'gs_fl')
        citelinks = [c for c in linksPaperNavigator if 'Citar' in c.text]
        for link in citelinks[:3]:
            citeElement = link.find_elements(By.TAG_NAME, 'a')[1]
            citeElement.click()
            time.sleep(5)
            citeFrame = webdriver.find_element(By.ID, 'gs_cit-bdy')
            citeTable = citeFrame.find_element(By.TAG_NAME, 'tbody')
            labels = [label.text for label in citeTable.find_elements(By.CLASS_NAME, 'gs_cith')]
            values = [val.text for val in citeTable.find_elements(By.CLASS_NAME, 'gs_citr')]
            collectionCiteFormats.append({key: values[idx] for idx, key in enumerate(labels)})
            closeButton = webdriver.find_element(By.ID, 'gs_cit-x')
            time.sleep(5)
            closeButton.click()
        logger.info(f'page #{num_page} loaded')
        return collectionCiteFormats
    except:
        logger.info(f'page #{num_page} not scraped, pass to the next paper.')
        return []


def rawDataToFrame(collection):
    df_metadata = pd.DataFrame(columns=list(collection[0].keys()))
    for item in collection:
        df_metadata = df_metadata.append(item, ignore_index=True)
    df_metadata.to_csv(data_folder / 'output' / 'data' / 'crawlerGSResult.csv', index=False)
    return


def loadingApp():
    logger.info('loading crawler application')
    url = 'https://scholar.google.com.br/'
    driver = webdriver.Chrome()
    driver.get(url)
    try:
        time.sleep(15)
        numResults = runQuery(webdriver=driver)
        pages = selectPages()
        return driver, numResults, pages
    except:
        driver.quit()
        logger.error(traceback.print_exc())
        raise Exception


def main():
    driver, numResults, url_pages = loadingApp()
    logger.info(f'#results: {numResults}')
    try:
        list_md = []
        for idx, url in enumerate(url_pages[:2]):
            logger.info(f'scraping page #{idx}/97')
            collection_metadata = infoCollection(webdriver=driver, url_page=url, num_page=idx)
            list_md = list_md + collection_metadata
            time.sleep(15)
        driver.quit()
        rawDataToFrame(list_md)
        logger.info("crawler process has been completed!")
    except:
        driver.quit()
        logger.error(traceback.print_exc())
        raise Exception


if __name__ == '__main__':
    main()
