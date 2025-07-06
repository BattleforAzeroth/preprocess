import json
import re
import sys
from tqdm import tqdm
from utils import csv
from pyverilog.vparser.lexer import dump_tokens


def tokenize(text):
    # 定义正则表达式模式
    # 匹配单词、单引号及其内容、单个特殊字符、缩进和换行符
    pattern = r'\b\w+\b|[^\w\s]|[\n\t]+'
    # 使用正则表达式进行分词
    tokens = re.findall(pattern, text)
    return tokens


def clean_quoted_strings(text):
    processed = re.sub(
        r'"([^"]*?)"',
        lambda m: '"' + re.sub(r'\s+', '', m.group(1)) + '"',
        text
    )

    # 重新分词（保护引号内容）
    return [x for x in re.split(r'("[^"]*")|\s+', processed) if x]


directives = (
    "`begin_keywords",
    "`celldefine",
    "`default_nettype",
    "`define",
    "`else",
    "`elsif",
    "`end_keywords",
    "`endcelldefine",
    "`endif",
    "`ifdef",
    "`ifndef",
    "`include",
    "`line",
    "`nounconnected_drive",
    "`pragma",
    "`resetall",
    "`timescale",
    "`unconnected_drive",
    "`undef",
    "`undefineall"
)

keywords = directives + (
    "1step", "accept_on", "alias", "always", "always_comb", "always_ff", "always_latch", "and", "assert", "assign",
    "assume", "automatic", "before", "begin", "bind", "bins", "binsof", "bit", "break", "buf", "bufif0", "bufif1",
    "byte", "case", "casex", "casez", "cell", "chandle", "checker", "class", "clocking", "cmos", "config", "const",
    "constraint", "context", "continue", "cover", "covergroup", "coverpoint", "cross", "deassign", "default",
    "defparam", "design", "disable", "dist", "do", "edge", "else", "end", "endcase", "endchecker", "endclass",
    "endclocking", "endconfig", "endfunction", "endgenerate", "endgroup", "endinterface", "endmodule", "endpackage",
    "endprimitive", "endprogram", "endproperty", "endspecify", "endsequence", "endtable", "endtask", "enum", "event",
    "eventually", "expect", "export", "extends", "extern", "final", "first_match", "for", "force", "foreach",
    "forever", "fork", "forkjoin", "function", "generate", "genvar", "global", "highz0", "highz1", "if", "iff",
    "ifnone", "ignore_bins", "illegal_bins", "implements", "implies", "import", "incdir", "include", "initial",
    "inout", "input", "inside", "instance", "int", "integer", "interconnect", "interface", "intersect", "join",
    "join_any", "join_none", "large", "let", "liblist", "library", "local", "localparam", "logic", "longint",
    "macromodule", "matches", "medium", "modport", "module", "nand", "negedge", "nettype", "new", "nexttime",
    "nmos", "nor", "noshowcancelled", "not", "notif0", "notif1", "null", "or", "output", "package", "packed",
    "parameter", "pmos", "posedge", "primitive", "priority", "program", "property", "protected", "pull0", "pull1",
    "pulldown", "pullup", "pulsestyle_ondetect", "pulsestyle_onevent", "pure", "rand", "randc", "randcase",
    "randsequence", "rcmos", "real", "realtime", "ref", "reg", "reject_on", "release", "repeat", "restrict",
    "return", "rnmos", "rpmos", "rtran", "rtranif0", "rtranif1", "s_always", "s_eventually", "s_nexttime",
    "s_until", "s_until_with", "scalared", "sequence", "shortint", "shortreal", "showcancelled", "signed", "small",
    "soft", "solve", "specify", "specparam", "static", "string", "strong", "strong0", "strong1", "struct", "super",
    "supply0", "supply1", "sync_accept_on", "sync_reject_on", "table", "tagged", "task", "this", "throughout",
    "time", "timeprecision", "timeunit", "tran", "tranif0", "tranif1", "tri", "tri0", "tri1", "triand", "trior",
    "trireg", "type", "typedef", "union", "unique", "unique0", "unsigned", "until", "until_with", "untyped", "use",
    "uwire", "var", "vectored", "virtual", "void", "wait", "wait_order", "wand", "weak", "weak0", "weak1", "while",
    "wildcard", "wire", "with", "within", "wor", "xnor", "xor"
)

operators = (
    "'", "'{", "{", "}", "[", "]", "(", ")", ";", ":", ":=", ":/", "::", ",", ".",
    "/", "*", "**", "*>", "+", "++", "+:", "+/-", "+%-", "-", "--", "-:", "->", "->>",
    "~", "~&", "~|", "~^", "$", "?", "#", "##", "#-#", "#=#", "^", "^~", "=", "==",
    "==?", "===", "=>", "+=", "-=", "/=", "*=", "&=", "|=", "%=", "^=", "<<=", "<<<=",
    ">>=", ">>>=", "<<", ">>", "<<<", ">>>", "!", "!=", "!=?", "!==", "%", "<", "<=",
    "<->", ">", ">=", "|", "||", "|->", "|=>", "@", "@@", "&", "&&", "&&&"
)

keywords_p = (
    'MODULE', 'ENDMODULE', 'BEGIN', 'END', 'GENERATE', 'ENDGENERATE', 'GENVAR',
    'FUNCTION', 'ENDFUNCTION', 'TASK', 'ENDTASK',
    'INPUT', 'INOUT', 'OUTPUT', 'TRI', 'REG', 'LOGIC', 'WIRE', 'INTEGER', 'REAL', 'SIGNED',
    'PARAMETER', 'LOCALPARAM', 'SUPPLY0', 'SUPPLY1',
    'ASSIGN', 'ALWAYS', 'ALWAYS_FF', 'ALWAYS_COMB', 'ALWAYS_LATCH', 'SENS_OR', 'POSEDGE', 'NEGEDGE', 'INITIAL',
    'IF', 'ELSE', 'FOR', 'WHILE', 'CASE', 'CASEX', 'CASEZ', 'UNIQUE', 'ENDCASE', 'DEFAULT',
    'WAIT', 'FOREVER', 'DISABLE', 'FORK', 'JOIN',
)

operators_p = (
    'PLUS', 'MINUS', 'POWER', 'TIMES', 'DIVIDE', 'MOD',
    'NOT', 'OR', 'NOR', 'AND', 'NAND', 'XOR', 'XNOR',
    'LOR', 'LAND', 'LNOT',
    'LSHIFTA', 'RSHIFTA', 'LSHIFT', 'RSHIFT',
    'LT', 'GT', 'LE', 'GE', 'EQ', 'NE', 'EQL', 'NEL',
    'COND',  # ?
    'EQUALS', 'AT', 'COMMA', 'COLON', 'SEMICOLON', 'DOT',
    'PLUSCOLON', 'MINUSCOLON',
    'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET', 'LBRACE', 'RBRACE',
    'DELAY', 'DOLLER',
)

string_p = ('STRING_LITERAL')

number_p = ('FLOATNUMBER',
            'INTNUMBER_DEC', 'SIGNED_INTNUMBER_DEC',
            'INTNUMBER_HEX', 'SIGNED_INTNUMBER_HEX',
            'INTNUMBER_OCT', 'SIGNED_INTNUMBER_OCT',
            'INTNUMBER_BIN', 'SIGNED_INTNUMBER_BIN',)

name_p = ('ID')

# 定义各数字格式的正则表达式
bin_number = r"[0-9]*'[bB][01xXzZ?][01xXzZ?_]*"
signed_bin_number = r"[0-9]*'[sS][bB][01xXzZ?][01xXzZ?_]*"
octal_number = r"[0-9]*'[oO][0-7xXzZ?][0-7xXzZ?_]*"
signed_octal_number = r"[0-9]*'[sS][oO][0-7xXzZ?][0-7xXzZ?_]*"
hex_number = r"[0-9]*'[hH][0-9a-fA-FxXzZ?][0-9a-fA-FxXzZ?_]*"
signed_hex_number = r"[0-9]*'[sS][hH][0-9a-fA-FxXzZ?][0-9a-fA-FxXzZ?_]*"

decimal_number = r"([0-9]*'[dD][0-9xXzZ?][0-9xXzZ?_]*)|([0-9][0-9_]*)"
signed_decimal_number = r"[0-9]*'[sS][dD][0-9xXzZ?][0-9xXzZ?_]*"

exponent_part = r"([eE][-+]?[0-9]+)"
fractional_constant = r"([0-9]*\.[0-9]+)|([0-9]+\.[0-9]*)"
float_number = fr"(({fractional_constant}{exponent_part}?)|([0-9]+{exponent_part}))"
# 组合所有数字格式的正则表达式
number_pattern = re.compile(
    r"^("
    fr"{bin_number}|{signed_bin_number}|"
    fr"{octal_number}|{signed_octal_number}|"
    fr"{hex_number}|{signed_hex_number}|"
    fr"{decimal_number}|{signed_decimal_number}|"
    fr"{float_number}"
    r")$"
)

if __name__ == '__main__':
    input_file = r'D:\source\Dataset\distinct.csv'
    output_file = "./out/all_tokenize_res.txt"
    df = csv.read_csv(input_file)
    # train_dict = {}
    # eval_dict = {}
    # test_dict = {}
    # token_seqs = []
    # token_sets = []
    dataset_dict = {}
    cnt = 0
    for no in tqdm(range(len(df)), desc="Parsing"):
        code = df["text"].iloc[no]
        code += '\n'

        result = []
        lines = code.splitlines()
        new_lines = []

        for lineno, line in enumerate(lines):
            if '`' in line:  # 如果该行包含反引号
                new_lines.append(f"//{line}")  # 在行前添加 //
                tokens = tokenize(line)
                text = ' '.join(tokens)  # 合并成字符串
                merged1 = re.sub(r'` (\w+)', r'`\1', text)  # 合并 ` 和后面的单词
                new_lst = clean_quoted_strings(merged1)

                for i, token in enumerate(new_lst):
                    if token in keywords:
                        result.append([token, 'KEYWORD', lineno, i])
                    elif token in operators:
                        result.append([token, 'OP', lineno, i])
                    elif len(token) >= 2 and token[0] == '"' and token[-1] == '"':
                        result.append([token, 'STRING', lineno, i])
                    elif number_pattern.fullmatch(token):
                        result.append([token, 'NUMBER', lineno, i])
                    else:
                        result.append([token, 'NAME', lineno, i])
                result.append(['\n', 'NEWLINE', lineno, len(new_lst)])
            else:
                new_lines.append(line)  # 否则保持不变

        # 重新组合成字符串
        new_text = '\n'.join(new_lines)
        try:
            # 可能报错的代码
            dump = dump_tokens(new_text)
        except Exception:  # 捕获所有异常
            continue  # 出错时跳过本次循环

        if not dump:
            continue
        first_line = dump.splitlines()[0].split()
        if len(first_line) > 4:
            first_part = ' '.join(first_line[:-3])
            first_line = [first_part] + first_line[-3:]

        current_lineno = int(first_line[2])
        for line in dump.splitlines():
            # print(line)
            lst = line.split()
            if len(lst) > 4:
                first_part = ' '.join(lst[:-3])
                lst = [first_part] + lst[-3:]
            if int(lst[2]) > current_lineno:
                result.append(['\n', 'NEWLINE', current_lineno, sys.maxsize])
                current_lineno = int(lst[2])
            if lst[1] in keywords_p:
                lst[1] = 'KEYWORD'
            elif lst[1] in operators_p:
                lst[1] = 'OP'
            elif lst[1] in string_p:
                lst[1] = 'STRING'
            elif lst[1] in number_p:
                lst[1] = 'NUMBER'
            elif lst[1] in name_p:
                lst[1] = 'NAME'
            result.append(lst)
        result = sorted(result, key=lambda x: (int(x[2]), int(x[3])))

        token_seq = [row[0] for row in result]

        # flag = True
        # for token in token_seqs:
        #     seq = ' '.join(token)
        #     current_seq = ' '.join(token_seq)
        #     # 相似度度量
        #     # if seq == current_seq:
        #     if xs.minhash(seq, current_seq) > 0.75:
        #         flag = False
        #
        # if flag:
        # token_seqs.append(token_seq)
        type_seq = [row[1] for row in result]

        # token_set = set(token_seq)
        # token_sets.append(token_set)
        # if cnt % 10 == 0:
        #     test_dict[cnt] = [token_seq, type_seq]
        # elif cnt % 10 < 3:
        #     eval_dict[cnt] = [token_seq, type_seq]
        # else:
        #     train_dict[cnt] = [token_seq, type_seq]
        dataset_dict[cnt] = [token_seq, type_seq]
        cnt += 1

    print(f"Parsed Dataset: {len(dataset_dict)}")
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(dataset_dict, file)
    # with open("eval_tokenize_res.txt", "w", encoding="utf-8") as file:
    #     json.dump(eval_dict, file)
    #
    # with open("train_tokenize_res.txt", "w", encoding="utf-8") as file:
    #     json.dump(train_dict, file)
    #
    # with open("test_tokenize_res.txt", "w", encoding="utf-8") as file:
    #     json.dump(test_dict, file)
