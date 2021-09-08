# create date: 2021/09/06
import xlrd


def save_label():
    workbook = xlrd.open_workbook('data/CASME2-coding-20140508.xlsx')

    sheet = workbook.sheets()[0]
    # 第B列
    print(len(sheet.col_values(8)[1:]), sheet.col_values(8)[1:])
    label_int_map = {
        "happiness": 1,

    }

def cap_lbp_feature():
    pass


def save_lbp_feature():
    pass


def load_lbp_feature():
    pass


if __name__ == '__main__':
    save_label()