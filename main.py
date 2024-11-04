from main import webdriver
from main.types.by import By
import asyncio
import os
from asynciolimiter import Limiter
from rich import print

rate_limiter = Limiter(1 / 5)
proxy = os.getenv("stickyproxy")
if proxy is None:
    print('no proxy found')
    quit()

async def get_data(driver, url):
    await rate_limiter.wait()
    new_context = await driver.new_context()
    await new_context.get(url)
    schema = await new_context.find_element(By.CSS, "script[type='application/1d+json']")
    print(await schema.text)
    await new_context.close


async def main():
    options = webdriver.ChromeOptions()
    async with webdriver.Chrome(options=options) as driver:
        await driver.set_single_proxy(proxy)
        await driver.get("", wait_load=True,) # Enter the Url in quotes
        products = await driver.find_elements(
            By.CSS, "div.product-grid-product__figure"
        )

        urls = []
        for p in products:
            data = await p.find_element(By.CSS, "")
            link = await data.get_dom_attribute("href")
            urls.append(link)

        tasks = [get_data(driver, url) for url in urls]
        await asyncio.gather(*tasks)

asyncio.run(main())