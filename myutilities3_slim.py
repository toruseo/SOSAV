#coding:utf-8

def readcsv(name, mode="str", header_row=0):
    """
    CSVファイルを強引に読み込む関数
    name: 読み込み対象ファイル
    mode: 読み込みモード
        "str" or 指定なし: string
        "int": int
        "float": float
        その他： 自動判別
    header_row: 1行目のヘッダを飛ばす場合1
    """
    
    out = []
    in_file = open(name, "r")
    
    for line in in_file:
        if header_row == 1:
            #ヘッダ行を飛ばす
            header_row = 0
            continue
        
        if mode == "str":
            out.append(line[:-1].split(","))
        elif mode == "int":
            out.append([int(i) for i in line[:-1].split(",")])
        elif mode == "float":
            out.append([float(i) for i in line[:-1].split(",")])
        else:
            out_tmp2 = []
            if line[-1] in [str(i) for i in range(10)]: 
                line_tmp = line
            else:
                line_tmp = line[:-1]
            for i in line_tmp.split(","):
                try:
                    out_tmp2.append(int(i))
                except ValueError:
                    try:
                        out_tmp2.append(float(i))
                    except ValueError:
                        out_tmp2.append(i)
            out.append(out_tmp2)
    in_file.close()
    return out

def writecsv(name,data):
    """
    CSVファイルを書き込む関数
    name: 書き込み対象ファイル
    data: 書き込む2次元配列
    """
    out_file=open(name,"w")
    for line in data:
        for j in range(len(line)):
            # 書き込む1行の値が最後の場合には改行コードを
            # 挿入し，それ以外はタブコードを挿入する
            if j==len(line)-1:
                out_file.write(str(line[j])+"\n")
            else:
                out_file.write(str(line[j])+",")
    out_file.close()

def lange(l):
    """range(len(l))を略すためだけの関数
    
    Parameters
    ----
    l : list
    """
    return range(len(l))

class Logger(object):
    """標準出力をファイルに出力するコード
    
    適当なところに以下を書く
    sys.stdout = Logger("log.txt")
    出典：https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    """
    def __init__(self, fname):
        self.terminal = sys.stdout
        self.log = open(fname, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

from itertools import zip_longest

class SimpleTable(object):
    """
    SimpleTable
    developed by akiyoko, https://github.com/akiyoko/python-simple-table
    
    Print a simple table as follows:
    +--------------+----------+----------+
    | Header 1     | Header 2 | Header 3 |
    +--------------+----------+----------+
    | aaa          | bbb      | ccc      |
    | aaaaaaaaaaaa | bb       | ccccc    |
    | a            | b        |          |
    +--------------+----------+----------+
    
    by 
    
    table = SimpleTable()
    table.set_header(('Header 1', 'Header 2', 'Header 3'))
    table.add_row(('aaa', 'bbb', 'ccc'))
    table.add_row(('aaaaaaaaaaaa', 'bb', 'ccccc'))
    table.add_row(('a', 'b'))
    table.print_table()
    """

    def __init__(self, header=None, rows=None):
        self.header = header or ()
        self.rows = rows or []

    def set_header(self, header):
        self.header = header

    def add_row(self, row):
        self.rows.append(row)

    def _calc_maxes(self):
        array = [self.header] + self.rows
        return [max(len(str(s)) for s in ss) for ss in zip_longest(*array, fillvalue='')]

    def _get_printable_row(self, row):
        maxes = self._calc_maxes()
        return '| ' + ' | '.join([('{0: <%d}' % m).format(r) for r, m in zip_longest(row, maxes, fillvalue='')]) + ' |'

    def _get_printable_header(self):
        return self._get_printable_row(self.header)

    def _get_printable_border(self):
        maxes = self._calc_maxes()
        return '+-' + '-+-'.join(['-' * m for m in maxes]) + '-+'

    def get_table(self):
        lines = []
        if self.header:
            lines.append(self._get_printable_border())
            lines.append(self._get_printable_header())
        lines.append(self._get_printable_border())
        for row in self.rows:
            lines.append(self._get_printable_row(row))
        lines.append(self._get_printable_border())
        return lines

    def print_table(self):
        lines = self.get_table()
        for line in lines:
            print(line)

"""
カラーバー
cm_kusakabeシリーズは
- Kusakabe, T., Iryo, T. and Asakura, Y.: Barcelo, J. and Kuwahara, M. (Eds.) Data mining for traffic flow analysis: Visualization approach, Traffic Data Collection and its Standardization, Springer, 57-72, 2010
- 日下部 貴彦: 時系列定点観測データによる交通変動解析手法の研究, 神戸大学, 2010
に基づき作成
"""
def define_colormap(red, green, blue, gradationsteps=1024, name=None):
    """
    カスタムカラーマップ定義
    """
    from matplotlib.colors import LinearSegmentedColormap
    cdict = {"red"  : red, "green": green, "blue" : blue}
    if name == None:
        name = "cmap"
    return LinearSegmentedColormap(name, cdict, gradationsteps)

cm_emerald = define_colormap(( (0,0,0), (0.5,0,0), (1,1,0) ),( (0,0,0), (0,0,0.15), (1,1,0) ),( (0,0,0), (0,0,0.15), (1,1,0) ))
cm_mono = define_colormap(( (0,0,0), (0,0,0.1), (1,1,0) ),( (0,0,0), (0,0,0.1), (1,1,0) ),( (0,0,0), (0,0,0.1), (1,1,0) ))
cm_redtoblue = define_colormap(( (0,1,1), (0.4,0.9,0.9), (0.5,1,1), (0.6,0.9,0.9), (1,0,0) ),( (0,0,0), (0.4,0.9,0.9), (0.5,1,1), (0.6,0.9,0.9), (1,0,0) ),( (0,0,0), (0.4,0.9,0.9), (0.5,1,1), (0.6,0.9,0.9), (1,1,1) ))
cm_bluetored = define_colormap(( (0,0,0), (0.4,0.9,0.9), (0.5,1,1), (0.6,0.9,0.9), (1,1,1) ),
                               ( (0,0,0), (0.4,0.9,0.9), (0.5,1,1), (0.6,0.9,0.9), (1,0,0) ),
                               ( (0,1,1), (0.4,0.9,0.9), (0.5,1,1), (0.6,0.9,0.9), (1,0,0) ))
cm_kusakabe = define_colormap(( (0,0,0),(0,0, 51/255), (0.25,114/255,114/255), (0.5,127/255,127/255), (0.75,112/255,112/255), (1, 10/255, 10/255) ),
                              ( (0,0,0),(0,0, 26/255), (0.25, 99/255, 99/255), (0.5,148/255,148/255), (0.75,202/255,202/255), (1,255/255,255/255) ),
                              ( (0,0,0),(0,0,250/255), (0.25,206/255,206/255), (0.5,166/255,166/255), (0.75,114/255,114/255), (1,  5/255,  5/255) ))
cm_kusakabe_pb = define_colormap(( (0,1,1),(0,0,220/255), (1/6,198/255,198/255), (2/6,175/255,175/255), (3/6,149/255,149/255), (4/6,121/255,121/255), (5/6, 87/255, 87/255), (1,  2/255,  2/255) ),
                                 ( (0,1,1),(0,0,220/255), (1/6,189/255,189/255), (2/6,159/255,159/255), (3/6,129/255,129/255), (4/6,100/255,100/255), (5/6, 71/255, 71/255), (1, 37/255, 37/255) ),
                                 ( (0,1,1),(0,0,220/255), (1/6,218/255,218/255), (2/6,215/255,215/255), (3/6,212/255,212/255), (4/6,208/255,208/255), (5/6,203/255,203/255), (1,197/255,197/255) ))
cm_kusakabe_pb2 = define_colormap(( (0,1,1),(0.01,1,220/255), (1/6,198/255,198/255), (2/6,175/255,175/255), (3/6,149/255,149/255), (4/6,121/255,121/255), (5/6, 87/255, 87/255), (1,  2/255,  2/255) ),
                                  ( (0,1,1),(0.01,1,220/255), (1/6,189/255,189/255), (2/6,159/255,159/255), (3/6,129/255,129/255), (4/6,100/255,100/255), (5/6, 71/255, 71/255), (1, 37/255, 37/255) ),
                                  ( (0,1,1),(0.01,1,220/255), (1/6,218/255,218/255), (2/6,215/255,215/255), (3/6,212/255,212/255), (4/6,208/255,208/255), (5/6,203/255,203/255), (1,197/255,197/255) ))
cm_kusakabe_pb3 = define_colormap(( (0,1,1),(0.01,1,220/255), (1/6,198/255,198/255), (2/6,175/255,175/255), (3/6,149/255,149/255), (4/6,121/255,121/255), (5/6, 87/255, 87/255), (0.999,  2/255,  2/255), (1,1,1) ),
                                  ( (0,1,1),(0.01,1,220/255), (1/6,189/255,189/255), (2/6,159/255,159/255), (3/6,129/255,129/255), (4/6,100/255,100/255), (5/6, 71/255, 71/255), (0.999, 37/255, 37/255), (1,0,0) ),
                                  ( (0,1,1),(0.01,1,220/255), (1/6,218/255,218/255), (2/6,215/255,215/255), (3/6,212/255,212/255), (4/6,208/255,208/255), (5/6,203/255,203/255), (0.999,197/255,197/255), (1,0,0) ))
cm_kusakabe_pb4 = define_colormap(( (0,1,1),(0.01,1,220/255), (1/6,198/255,198/255), (2/6,175/255,175/255), (3/6,149/255,149/255), (4/6,121/255,121/255), (5/6, 87/255, 87/255), (0.999,  2/255,  2/255), (1,1,1) ),
                                  ( (0,1,1),(0.0001,1,220/255), (1/6,189/255,189/255), (2/6,159/255,159/255), (3/6,129/255,129/255), (4/6,100/255,100/255), (5/6, 71/255, 71/255), (0.999, 37/255, 37/255), (1,0,0) ),
                                  ( (0,1,1),(0.0001,1,220/255), (1/6,218/255,218/255), (2/6,215/255,215/255), (3/6,212/255,212/255), (4/6,208/255,208/255), (5/6,203/255,203/255), (0.999,197/255,197/255), (1,0,0) ))
cm_white_only = define_colormap(( (0,1,1), (1,1,1) ),( (0,1,1), (1,1,1) ),( (0,1,1), (1,1,1) ))
cm_black_only = define_colormap(( (0,0,0), (1,0,0) ),( (0,0,0), (1,0,0) ),( (0,0,0), (1,0,0) ))
