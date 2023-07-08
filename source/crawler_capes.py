import warnings

warnings.filterwarnings("ignore")

import traceback
import pandas as pd
import time
from pathlib import Path
import os


data_folder = Path(os.path.dirname(__file__))


import logging


logfile = data_folder / 'logs' / 'crawler_capes' /f"{pd.to_datetime('today').strftime('%b %d %Y %I:%M%p')}.log"
logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    filemode="w",
                    level=logging.INFO,
                    filename=logfile)
logger = logging.getLogger()


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


DICT_DEFAULT = {
    'type': 'N/I'
    , 'Título': 'N/I'
    , 'Autor': 'N/I'
    , 'Assuntos': 'N/I'
    , 'É parte de': 'N/I'
    , 'Descrição': 'N/I'
    , 'Editor': 'N/I'
    , 'Idioma': 'N/I'
    , 'Identificador': 'N/I'
    , 'Fonte': 'N/I'
    , 'reference': 'N/I'
    , 'availableFullText': 'N/I'
}


def get_pages(wdriver, pos=None):
    logger.info('selecting page objetcs')
    navigationBar = wdriver.find_element_by_class_name("counter-main")
    time.sleep(10)
    pages = navigationBar.find_elements_by_tag_name('a')
    if pos:
        pages = pages[pos:]
    dict_pages = {item.text: item for item in pages}
    count = 1
    while True:
        try:
            expandResults = wdriver.find_element(By.XPATH, "//*[contains(text(), 'Expandir meus resultados')]")
            expandResults.click()
            break
        except:
            logger.info(f"Expanding results failed #{count}")
            count += 1
            time.sleep(5)
    _ = WebDriverWait(wdriver, 30).until(EC.visibility_of_element_located((By.CLASS_NAME, 'results-count')))
    num_results = int(wdriver.find_elements_by_class_name('results-count')[1].text.split(' ')[0].replace('.', ''))
    return dict_pages, num_results


def runQuery(wdriver):
    logger.info('running query search')
    # time.sleep(30)
    keywords = '''"data science platform" OR "data science framework" OR "data science model"'''
    #  ("small and medium sized enterprize" OR "SME") AND ("information systems" OR "information technology" OR "information management")
    searchBar = wdriver.find_element_by_id("searchBar")
    searchBar.clear()
    searchBar.send_keys(keywords)
    searchBar.submit()
    _ = WebDriverWait(wdriver, 60).until(EC.visibility_of_element_located((By.CLASS_NAME, 'counter-main')))
    return get_pages(wdriver, pos=None)


def openFirstPaper(wdriver):
    _ = WebDriverWait(wdriver, 60).until(EC.visibility_of_element_located((By.CLASS_NAME, 'item-title')))
    paperFrames = wdriver.find_elements(By.CLASS_NAME, 'item-title')
    logger.info(f'number of papers: {len(paperFrames)}')
    initialPaper = paperFrames[0]
    time.sleep(25)
    initialPaper.click()
    time.sleep(15)
    return


def infoCollection(wdriver, num_paper):
    try:
        time.sleep(10)
        dict_paper_metadata = {}
        _ = WebDriverWait(wdriver, 20).until(EC.visibility_of_element_located((By.CLASS_NAME
                                                                               , 'full-view-inner-container')))
        type_paper = wdriver.find_element(By.CLASS_NAME, "full-view-inner-container").\
                             find_element(By.CLASS_NAME, "media-content-type").text
        dict_paper_metadata['type'] = type_paper
        fulltext = wdriver.find_element(By.CLASS_NAME, "full-view-inner-container").\
                            find_element(By.CLASS_NAME, "button-content").text
        dict_paper_metadata['availableFullText'] = fulltext
        _ = WebDriverWait(wdriver, 20).until(EC.visibility_of_element_located((By.CLASS_NAME, 'flex-gt-xs-25')))
        detailLabels = [l.text for l in wdriver.find_elements(By.CLASS_NAME, "flex-gt-xs-25")]
        detailValues = [v.text for v in wdriver.find_elements(By.CLASS_NAME, "item-details-element")]
        if len(detailValues) != len(detailLabels):
            logger.info(50 * '-' + 'LABELS NOT MATCHED VALUES LIST' + 50 * '-')
            logger.info(f'{len(detailLabels)}' + '|' + f'{len(detailLabels)}')
        if len(detailValues) == len(detailLabels) + 1:
            detailValues[1] = detailValues[1] + ';' + detailValues[2]
            del detailValues[2]
        for idx, val in enumerate(detailLabels):
            dict_paper_metadata[val] = detailValues[idx]
        logger.info(f'loaded: {type_paper}\n{detailValues[0]}')
        return dict_paper_metadata
    except:
        logger.info(f'#{num_paper} not scraped, pass to the next paper.')
        return DICT_DEFAULT


def rawDataToFrame(collection):
    output_path = data_folder / 'output' / 'data' / f"RawCapesCrawlerResult{pd.to_datetime('today').strftime('%b%d%Y%I%M%p')}.csv"
    df_metadata = pd.DataFrame(collection)
    df_metadata.to_csv(output_path, index=False)
    return


def loadingApp():
    logger.info('loading crawler application')
    url = 'https://capes-primo.ezl.periodicos.capes.gov.br/primo-explore/search?vid=CAPES_V3&amp;lang=pt_BR&amp;tab=default_tab&amp;search_scope=default_scope&amp;offset=0&amp;'
    driver = webdriver.Chrome()
    driver.get(url)
    try:
        time.sleep(10)
        dict_page_objects, num_results = runQuery(wdriver=driver)
        time.sleep(10)
        openFirstPaper(wdriver=driver)
        return dict_page_objects, num_results, driver
    except:
        driver.quit()
        logger.error(traceback.format_exc())
        raise Exception


def main():
    dict_page_objects, num_results, driver = loadingApp()
    nPaper = 1
    list_dict_md = []
    try:
        while nPaper <= num_results:
            logger.info(f'scraping paper #{nPaper}/{num_results}')
            dict_paper_metadata = infoCollection(wdriver=driver, num_paper=nPaper)
            _ = WebDriverWait(driver, 200).until(EC.visibility_of_element_located((By.ID, 'CitationButtonFullView')))
            driver.find_element(By.ID, "CitationButtonFullView").click()
            time.sleep(3)
            reference = driver.find_element(By.CLASS_NAME, "form-focus").text
            dict_paper_metadata['reference'] = reference
            list_dict_md.append(dict_paper_metadata)
            while True:
                c = 1
                try:
                    driver.find_elements(By.CLASS_NAME, "close-button")[-1].click()
                    break
                except:
                    logger.info(f"Trying to pass to the next paper: #{c}")
                    c += 1
                    time.sleep(2)
            nPaper += 1
        driver.quit()
        rawDataToFrame(list_dict_md)
        logger.info("crawler process has been completed!")
    except:
        logger.error(traceback.format_exc())
        raise Exception


if __name__ == '__main__':
    main()
