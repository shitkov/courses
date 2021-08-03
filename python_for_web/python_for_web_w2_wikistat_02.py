import os
import re

from bs4 import BeautifulSoup


def parse(path_to_file):
    with open(path_to_file, encoding="utf-8") as file:
        soup = BeautifulSoup(file, "lxml")
        body = soup.find(id="bodyContent")

    imgs = len(body.find_all('img', width=lambda x: int(x or 0) > 199))

    headers = sum(1 for tag in body.find_all(
        ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']) if tag.get_text()[0] in "ETC")

    lists = sum(
        1 for tag in body.find_all(['ol', 'ul']) if not tag.find_parent(['ol', 'ul']))

    linkslen = 0

    for a in body.find_all('a'):
        current_streak = 1

        for tag in a.find_next_siblings():
            if tag.name == 'a':
                current_streak += 1
            else:
                break

        linkslen = current_streak if current_streak > linkslen else linkslen

    return [imgs, headers, linkslen, lists]


def get_statistics(path, start_page, end_page):
    """собирает статистику со страниц, возвращает словарь, где ключ - название страницы,
    значение - список со статистикой страницы"""
    pages = build_bridge(path, start_page, end_page)
    statistic = {}

    for page in pages:
        statistic[page] = parse(os.path.join(path, page))

    return statistic


def get_links(path, page):
    """возвращает множество названий страниц, ссылки на которые содержатся в файле page"""

    with open(os.path.join(path, page), encoding="utf-8") as file:
        links = set(re.findall(r"(?<=/wiki/)[\w()]+", file.read()))
        if page in links:
            links.remove(page)
    return links


def get_backlinks(path, end_page, unchecked_pages, checked_pages, backlinks):
    """возвращает словарь обратных ссылок (ключ - страница, значение - страница
    с которой возможен переход по ссылке на страницу, указанную в ключе)"""

    if end_page in checked_pages or not checked_pages:
        return backlinks

    new_checked_pages = set()

    for checked_page in checked_pages:
        unchecked_pages.remove(checked_page)
        linked_pages = get_links(path, checked_page) & unchecked_pages

        for linked_page in linked_pages:
            backlinks[linked_page] = backlinks.get(linked_page, checked_page)
            new_checked_pages.add(linked_page)

    checked_pages = new_checked_pages & unchecked_pages

    return get_backlinks(path, end_page, unchecked_pages, checked_pages, backlinks)


def build_bridge(path, start_page, end_page):
    """возвращает список страниц, по которым можно перейти по ссылкам со start_page на
    end_page, начальная и конечная страницы включаются в результирующий список"""

    backlinks = \
        get_backlinks(path, end_page, set(os.listdir(path)), {start_page, }, dict())

    current_page, bridge = end_page, [end_page]

    while current_page != start_page:
        current_page = backlinks.get(current_page)
        bridge.append(current_page)

    return bridge[::-1]