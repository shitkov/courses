from bs4 import BeautifulSoup
import unittest
import re

def get_link_count(links_list):
    max_len = 1

    for i in range(len(links_list)):
        if str(links_list[i])[1] == 'a':
            max_len += 1
        else:
            break

    return max_len


def parse(path_to_file):
    # imgs
    with open(path_to_file, encoding='utf-8') as f:
        read_data = f.read()
    
    soup = BeautifulSoup(read_data, 'lxml')
    body = soup.find(id="bodyContent")
    exp = r' width=\"(\d*)'
    match = re.findall(exp, str(body.find_all('img')))
    match = [int(i) for i in match]
    imgs = len([i for i in match if i >= 200])

    # headers
    headers_list = body.find_all(name=re.compile('h[1-6]'))
    headers = 0

    for header_current in headers_list:
        tag_string = header_current.get_text()
        headers += 1 if re.search('^[ETC]', tag_string) else 0
        
    # linkslen
    linkslen = 0

    tags = body('a')
    for tag in tags:
        tmp_len = get_link_count(tag.find_next_siblings())
        if tmp_len > linkslen:
            linkslen = tmp_len
    
    # lists
    lists = 0
    tags = body.find_all(['ul', 'ol'])

    for tag in tags:
        if not tag.find_parents(['ul', 'ol']):
            lists += 1

    return [imgs, headers, linkslen, lists]

class TestParse(unittest.TestCase):
    def test_parse(self):
        test_cases = (
            ('wiki/Stone_Age', [13, 10, 12, 40]),
            ('wiki/Brain', [19, 5, 25, 11]),
            ('wiki/Artificial_intelligence', [8, 19, 13, 198]),
            ('wiki/Python_(programming_language)', [2, 5, 17, 41]),
            ('wiki/Spectrogram', [1, 2, 4, 7]),)

        for path, expected in test_cases:
            with self.subTest(path=path, expected=expected):
                self.assertEqual(parse(path), expected)


if __name__ == '__main__':
    unittest.main()
