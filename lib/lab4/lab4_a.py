import re

import requests
from bs4 import BeautifulSoup
from mpi4py import MPI
from transliterate import translit


# (Measure-Command {mpiexec -n 6 python lb4/lb4_a.py}).toString()
#  time mpiexec -n 6 python lb4/lb4_a.py


def is_year(str):
    return str.isnumeric() and len(str) == 4 and (str.startswith("1") or str.startswith("20"))


def parsing(url):
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    article_text = soup.find('div', {'id': 'content'}).get_text()
    article_text = article_text[:article_text.find("Примечания[править | править код]")]
    reals = [m.group().replace("\xa0", "") for m in re.finditer(r'-?\d+((\s\d+)+|(,\d+)?)', article_text)]
    integers = []
    years = []
    for num in reals:
        if not ',' in num and not is_year(num):
            integers.append(num)
        if is_year(num):
            years.append(num)
    return f"Reals {len(reals)}, ints {len(integers)}, years {len(years)} {translit(soup.title.get_text(), 'ru', reversed=True)}"


if __name__ == "__main__":
    urls = ["https://ru.wikipedia.org/wiki/%D0%A1%D0%B0%D1%85%D0%B0%D0%BB%D0%B8%D0%BD",
            "https://ru.wikipedia.org/wiki/%D0%9A%D0%BE%D0%BA%D0%BE%D1%81_(%D0%BE%D1%81%D1%82%D1%80%D0%BE%D0%B2)",
            "https://ru.wikipedia.org/wiki/%D0%9E%D1%81%D1%82%D1%80%D0%BE%D0%B2_%D0%9F%D0%B0%D1%81%D1 % 85 % D0 % B8",
            "https://ru.wikipedia.org/wiki/%D0%A0%D0%B5%D1%8E%D0%BD%D1%8C%D0%BE%D0%BD_(%D0%BE%D1 % 81 % D1 % 82 % D1 % 80 % D0 % BE % D0 % B2)",
            "https://ru.wikipedia.org/wiki/%D0%A5%D0%BE%D0%BA%D0%BA%D0%B0%D0%B9%D0%B4%D0%BE",
            "https://ru.wikipedia.org/wiki/%D0%9E%D1%81%D1%82%D1%80%D0%BE%D0%B2_%D0%A1%D0%B2%D1% 8F % D1 % 82 % D0 % BE % D0 % B9_ % D0 % 95 % D0 % BB % D0 % B5 % D0 % BD % D1 % 8B"]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

urls_per_process = len(urls) // size
start_index = rank * urls_per_process
end_index = start_index + urls_per_process if rank < size - 1 else len(urls)
urls_to_process = urls[start_index:end_index]

start = MPI.Wtime()
res = []
for url in urls_to_process:
    res.append(parsing(url))
finish = MPI.Wtime()
print(rank, finish - start)
for r in res:
    print(rank, r)
