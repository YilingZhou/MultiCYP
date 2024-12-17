import ast
import sys
import pkg_resources

# 标准库列表
STDLIB_MODULES = {
    'abc', 'aifc', 'argparse', 'array', 'ast', 'asyncio', 'base64',
    'binascii', 'builtins', 'cgi', 'collections', 'copy', 'csv',
    'datetime', 'decimal', 'difflib', 'enum', 'fractions', 'functools',
    'gc', 'getopt', 'getpass', 'gettext', 'glob', 'gzip', 'hashlib',
    'heapq', 'hmac', 'html', 'http', 'imaplib', 'imp', 'importlib',
    'inspect', 'io', 'itertools', 'json', 'logging', 'math', 'mimetypes',
    'multiprocessing', 'netrc', 'numbers', 'operator', 'os', 'pathlib',
    'pickle', 'pkgutil', 'platform', 'pprint', 'py_compile', 'queue',
    'random', 're', 'reprlib', 'secrets', 'select', 'shelve', 'shlex',
    'shutil', 'signal', 'socket', 'socketserver', 'sqlite3', 'ssl',
    'stat', 'string', 'struct', 'subprocess', 'sys', 'tempfile',
    'threading', 'time', 'timeit', 'token', 'tokenize', 'traceback',
    'typing', 'unittest', 'urllib', 'uuid', 'warnings', 'wave',
    'weakref', 'webbrowser', 'xml', 'xmlrpc', 'zipfile', 'zlib'
}


def get_imports(file_path):
    """分析Python文件中的导入语句"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    imports = set()
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except Exception as e:
        print(f"解析错误: {e}")

    return imports


def get_package_version(package_name):
    """获取已安装包的版本"""
    try:
        return pkg_resources.get_distribution(package_name).version
    except:
        return None


def generate_requirements(file_path):
    """生成requirements.txt文件"""
    # 获取导入的包
    imports = get_imports(file_path)

    # 过滤掉标准库
    third_party = {imp for imp in imports if imp not in STDLIB_MODULES}

    # 写入文件
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        for package in sorted(third_party):
            version = get_package_version(package)
            if version:
                f.write(f"{package}=={version}\n")
            else:
                print(f"警告: 未找到包 {package} 的版本信息")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python script.py <python文件路径>")
        sys.exit(1)

    file_path = sys.argv[1]
    generate_requirements(file_path)
    print("requirements.txt 已生成")