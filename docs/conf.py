# Configuration file for the Sphinx documentation builder

import glob
import inspect
import importlib
import re
import subprocess
import dased

## Project

package = 'dased'
project = 'dased'
version = dased.__version__
copyright = '2022-2023, Dominik Strutz'
repository = 'https://github.com/dominik-strutz/dased'
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()

## Extensions

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
    'myst_nb',
    'sphinx_design',
]

# MyST-NB configuration
nb_execution_mode = 'off'  # Don't execute notebooks during build
nb_execution_allow_errors = True  # Allow notebooks with errors to be included
nb_execution_excludepatterns = ['_build/**', 'jupyter_execute/**']  # Prevent recursion
nb_execution_in_temp = True  # Execute in temporary directory

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__call__',

}
autodoc_inherit_docstrings = False
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
autodoc_typehints_format = 'short'

autosummary_ignore_module_all = False

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable', None),
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
}

def linkcode_resolve(domain: str, info: dict) -> str:
    module = info.get('module', '')
    fullname = info.get('fullname', '')

    if not module or not fullname:
        return None

    objct = importlib.import_module(module)
    for name in fullname.split('.'):
        objct = getattr(objct, name)

    try:
        file = inspect.getsourcefile(objct)
        file = file[file.rindex(package) :]

        lines, start = inspect.getsourcelines(objct)
        end = start + len(lines) - 1
    except Exception as e:
        return None
    else:
        return f'{repository}/blob/{commit}/{file}#L{start}-L{end}'


napoleon_custom_sections = [
    ('Shapes', 'params_style'),
    'Wikipedia',
]

## Settings

add_function_parentheses = False
default_role = 'literal'
exclude_patterns = ['templates', '_build', 'jupyter_execute', '**/jupyter_execute/**']
html_copy_source = False
html_css_files = [
    'custom.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
]
# html_favicon = 'static/logo.svg'
# html_logo = 'static/logo.svg'
html_show_sourcelink = False
html_sourcelink_suffix = ''
html_static_path = ['static']
html_theme = 'furo'
html_theme_options = {
    'footer_icons': [
        {
            'name': 'GitHub',
            'url': repository,
            'html': '<i class="fa-brands fa-github fa-lg"></i>',
            'class': '',
        },
    ],
    'light_css_variables': {
        'color-api-keyword': "#000000",
        'color-api-name': "#000000",
        'color-api-pre-name': "#000000",
    },
    'dark_css_variables': {
        'color-api-keyword': "#000000",
        'color-api-name': "#000000",
        'color-api-pre-name': "#000000",
    },
    'sidebar_hide_name': True,
}
html_title = f'{project} {version}'
pygments_style = 'sphinx'
pygments_dark_style = 'monokai'
rst_prolog = """
.. role:: py(code)
    :class: highlight
    :language: python
"""
templates_path = ['templates']

## Edit HTML

def edit_html(app, exception):
    if exception:
        raise exception

    for file in glob.glob(f'{app.outdir}/**/*.html', recursive=True):
        with open(file, 'r') as f:
            text = f.read()

        text = text.replace('<a class="muted-link" href="https://pradyunsg.me">@pradyunsg</a>\'s', '')
        text = text.replace('<span class="pre">[source]</span>', '<i class="fa-solid fa-code"></i>')
        text = re.sub(r'(<a class="reference external".*</a>)(<a class="headerlink".*</a>)', r'\2\1', text)

        with open(file, 'w') as f:
            f.write(text)

def setup(app):
    app.add_css_file('custom.css')
    app.connect('build-finished', edit_html)