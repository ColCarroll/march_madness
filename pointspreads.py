import requests

START_YEAR = 2005
END_YEAR = 2015


def get_pointspread_url(year):
    if len(str(year)) == 4:
        year = str(year)[-2:]
    year = int(year)
    if year not in range(5, 15):
        raise ValueError("Pointspreads only exist between 2005-2015")
    if year == 14:
        return "http://www.thepredictiontracker.com/ncaabb14.csv"
    return "http://home.comcast.net/~tlbeck/ncaabb{:02d}.csv".format(year)


def get_season_data(year):
    r = requests.get(get_pointspread_url(year))
    return r.text


def get_all_seasons():
    data = []
    for year in range(START_YEAR, END_YEAR):
        print("Querying {:d}".format(year))
        year_data = get_season_data(year).splitlines()
        if data:
            year_data = year_data[1:]
        data.extend(year_data)
    return "\n".join(data)


def main():
    pass


if __name__ == '__main__':
    main()
