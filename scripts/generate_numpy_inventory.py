#!/usr/bin/env python3
"""
Comprehensive NumPy Function Inventory Generator

This script discovers and catalogs all NumPy functions across all modules,
creating a complete inventory for comparison with the rust-numpy implementation.
"""

import numpy as np
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json


def get_numpy_functions() -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Discover all NumPy functions organized by module.
    
    Returns:
        Dict mapping module names to list of (function_name, signature, docstring) tuples
    """
    modules_to_scan = [
        # Core array creation and manipulation
        'numpy', 'numpy.core', 'numpy.core.multiarray',
        'numpy.lib', 'numpy.lib.index_tricks',
        
        # Mathematical operations
        'numpy.fft', 'numpy.linalg', 'numpy.random',
        'numpy.polynomial', 'numpy.math', 'numpy.emath',
        
        # Data types and casting
        'numpy.dtype', 'numpy.core.numerictypes',
        
        # Array operations
        'numpy.core.arrayprint', 'numpy.core.fromnumeric',
        'numpy.core.defchararray', 'numpy.core.records',
        
        # Advanced functionality
        'numpy.ma', 'numpy.matlib', 'numpy.ctypeslib',
        'numpy.distutils', 'numpy.f2py',
        
        # Constants and special functions
        'numpy.lib.scimath', 'numpy.lib.stride_tricks',
        
        # IO and operations
        'numpy.io', 'numpy.lib.format', 'numpy.lib.npyio',
        'numpy.lib.function_base', 'numpy.lib.nanfunctions',
        'numpy.lib.shape_base', 'numpy.lib.type_check',
        'numpy.lib.arraysetops', 'numpy.lib.arraypad',
        'numpy.lib.histograms', 'numpy.lib.twodim_base',
        'numpy.lib.ufunclike',
    ]
    
    function_inventory = {}
    
    for module_name in modules_to_scan:
        try:
            module = __import__(module_name, fromlist=[''])
            functions = get_module_functions(module, module_name)
            if functions:
                function_inventory[module_name] = functions
                print(f"Found {len(functions)} functions in {module_name}")
        except ImportError as e:
            print(f"Could not import {module_name}: {e}")
        except Exception as e:
            print(f"Error scanning {module_name}: {e}")
    
    return function_inventory


def get_module_functions(module, module_name: str) -> List[Tuple[str, str, str]]:
    """Extract all callable functions from a module."""
    functions = []
    
    for name in dir(module):
        obj = getattr(module, name)
        
        # Skip private attributes and modules
        if name.startswith('_'):
            continue
        if inspect.ismodule(obj):
            continue
            
        # Check if it's a function or ufunc
        if (inspect.isfunction(obj) or 
            inspect.isbuiltin(obj) or 
            str(type(obj).__name__ in ['ufunc', 'function']):
            
            try:
                signature = str(inspect.signature(obj)) if inspect.isfunction(obj) else "ufunc"
                docstring = inspect.getdoc(obj) or "No documentation"
                
                # Truncate long docstrings
                if len(docstring) > 200:
                    docstring = docstring[:200] + "..."
                
                functions.append((name, signature, docstring))
            except Exception as e:
                # Some functions might not be inspectable
                functions.append((name, "non-inspectable", ""))
    
    return functions


def categorize_functions(function_inventory: Dict[str, List[Tuple[str, str, str]]]) -> Dict[str, List[str]]:
    """Categorize functions by functional area."""
    categories = {
        'array_creation': [],
        'array_manipulation': [],
        'mathematical': [],
        'linear_algebra': [],
        'statistics': [],
        'fft': [],
        'random': [],
        'io': [],
        'string': [],
        'datetime': [],
        'comparison': [],
        'logical': [],
        'bitwise': [],
        'dtype': [],
        'indexing': [],
        'broadcasting': [],
        'constants': [],
        'utility': [],
        'advanced': []
    }
    
    # Keywords for categorization
    category_keywords = {
        'array_creation': ['array', 'zeros', 'ones', 'empty', 'full', 'arange', 'linspace', 
                        'logspace', 'geomspace', 'eye', 'identity', 'from', 'load', 'gen'],
        'array_manipulation': ['reshape', 'transpose', 'flatten', 'ravel', 'squeeze', 'expand_dims',
                           'concatenate', 'stack', 'hstack', 'vstack', 'dstack', 'column_stack',
                           'split', 'array_split', 'hsplit', 'vsplit', 'dsplit', 'tile', 'repeat',
                           'sort', 'argsort', 'lexsort', 'searchsorted', 'unique'],
        'mathematical': ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'power', 'abs', 'sign',
                       'floor', 'ceil', 'round', 'trunc', 'add', 'subtract', 'multiply', 'divide'],
        'linear_algebra': ['dot', 'matmul', 'inner', 'outer', 'tensordot', 'inv', 'pinv', 'det',
                         'eig', 'svd', 'qr', 'solve', 'lstsq', 'norm', 'rank', 'trace'],
        'statistics': ['mean', 'var', 'std', 'corrcoef', 'cov', 'median', 'percentile', 'quantile',
                    'sum', 'prod', 'cumsum', 'cumprod', 'min', 'max', 'amin', 'amax', 'ptp'],
        'fft': ['fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn', 'rfft', 'irfft', 'fftshift'],
        'random': ['random', 'rand', 'randn', 'randint', 'choice', 'shuffle', 'permutation',
                  'uniform', 'normal', 'binomial', 'poisson', 'exponential'],
        'io': ['load', 'save', 'savez', 'savez_compressed', 'loadtxt', 'savetxt', 'fromfile', 'tofile'],
        'string': ['char', 'string', 'add', 'multiply', 'center', 'ljust', 'rjust', 'zfill'],
        'datetime': ['datetime64', 'timedelta64', 'busday', 'is_busday'],
        'comparison': ['greater', 'less', 'equal', 'not_equal', 'greater_equal', 'less_equal',
                    'maximum', 'minimum', 'fmax', 'fmin'],
        'logical': ['logical_and', 'logical_or', 'logical_not', 'logical_xor', 'isfinite', 'isinf',
                  'isnan', 'isneginf', 'isposinf'],
        'bitwise': ['bitwise_and', 'bitwise_or', 'bitwise_xor', 'invert', 'left_shift', 'right_shift'],
        'dtype': ['dtype', 'can_cast', 'common_type', 'min_scalar_type', 'result_type'],
        'indexing': ['take', 'put', 'choose', 'where', 'extract', 'place', 'copyto'],
        'broadcasting': ['broadcast', 'broadcast_arrays', 'broadcast_to'],
        'constants': ['pi', 'e', 'inf', 'nan', 'newaxis', 'NaN', 'Infinity'],
    }
    
    # Categorize functions based on names and documentation
    for module_name, functions in function_inventory.items():
        for func_name, signature, docstring in functions:
            func_lower = func_name.lower()
            categorized = False
            
            for category, keywords in category_keywords.items():
                if any(keyword in func_lower for keyword in keywords):
                    categories[category].append(func_name)
                    categorized = True
                    break
            
            if not categorized:
                categories['utility'].append(func_name)
    
    # Remove duplicates and sort
    for category in categories:
        categories[category] = sorted(list(set(categories[category])))
    
    return categories


def generate_markdown_report(function_inventory: Dict, categories: Dict, output_path: Path):
    """Generate comprehensive markdown report."""
    
    total_functions = sum(len(functions) for functions in function_inventory.values())
    
    with open(output_path, 'w') as f:
        f.write("# Comprehensive NumPy Function Inventory\n\n")
        f.write(f"**Generated:** {np.__version__}\n")
        f.write(f"**Total Functions:** {total_functions}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This inventory catalogs all NumPy functions across all modules for the purpose of ")
        f.write("achieving 100% API parity with the rust-numpy implementation.\n\n")
        
        # Module Breakdown
        f.write("## Module Breakdown\n\n")
        f.write("| Module | Function Count | Status |\n")
        f.write("|---------|----------------|--------|\n")
        
        for module_name, functions in sorted(function_inventory.items()):
            f.write(f"| `{module_name}` | {len(functions)} | 📋 To Analyze |\n")
        
        # Categorized Breakdown
        f.write("\n## Functional Categories\n\n")
        f.write("| Category | Function Count | Priority |\n")
        f.write("|----------|----------------|----------|\n")
        
        for category, functions in sorted(categories.items()):
            priority = "HIGH" if category in ['array_creation', 'array_manipulation', 'mathematical', 'linear_algebra'] else \
                     "MEDIUM" if category in ['statistics', 'fft', 'random', 'io'] else "LOW"
            f.write(f"| {category.replace('_', ' ').title()} | {len(functions)} | {priority} |\n")
        
        # Detailed Function Lists
        f.write("\n## Detailed Function Lists\n\n")
        
        for category, functions in sorted(categories.items()):
            if not functions:
                continue
                
            f.write(f"### {category.replace('_', ' ').title()}\n\n")
            
            for func_name in functions[:20]:  # Show first 20
                f.write(f"- `{func_name}`\n")
            
            if len(functions) > 20:
                f.write(f"- ... and {len(functions) - 20} more functions\n")
            
            f.write(f"**Total: {len(functions)} functions**\n\n")
        
        # Full Inventory Appendix
        f.write("## Appendix: Complete Module Inventories\n\n")
        for module_name, functions in sorted(function_inventory.items()):
            if not functions:
                continue
                
            f.write(f"### {module_name}\n\n")
            f.write("| Function | Signature | Documentation |\n")
            f.write("|----------|-----------|----------------|\n")
            
            for func_name, signature, docstring in functions:
                # Escape markdown special characters
                safe_signature = signature.replace('|', '\\|').replace('\n', ' ')
                safe_docstring = (docstring.replace('|', '\\|').replace('\n', ' ')[:100])
                
                f.write(f"| `{func_name}` | `{safe_signature}` | {safe_docstring}... |\n")
            
            f.write("\n")


def save_json_data(function_inventory: Dict, categories: Dict, output_path: Path):
    """Save raw data as JSON for further processing."""
    data = {
        'inventory': function_inventory,
        'categories': categories,
        'metadata': {
            'numpy_version': np.__version__,
            'generation_time': str(np.datetime64('now')),
            'total_functions': sum(len(funcs) for funcs in function_inventory.values())
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def main():
    """Main execution function."""
    print("NumPy Function Inventory Generator")
    print("=" * 40)
    print(f"NumPy version: {np.__version__}")
    print()
    
    # Create output directory
    output_dir = Path('scripts')
    output_dir.mkdir(exist_ok=True)
    
    print("Discovering NumPy functions...")
    function_inventory = get_numpy_functions()
    
    print("Categorizing functions...")
    categories = categorize_functions(function_inventory)
    
    # Generate reports
    markdown_path = output_dir / 'numpy_function_inventory.md'
    json_path = output_dir / 'numpy_function_inventory.json'
    
    print(f"Generating markdown report: {markdown_path}")
    generate_markdown_report(function_inventory, categories, markdown_path)
    
    print(f"Saving JSON data: {json_path}")
    save_json_data(function_inventory, categories, json_path)
    
    print()
    print("Summary:")
    total_functions = sum(len(funcs) for funcs in function_inventory.values())
    print(f"  Total modules found: {len(function_inventory)}")
    print(f"  Total functions found: {total_functions}")
    print(f"  Categories identified: {len([c for c in categories.values() if c])}")
    
    print()
    print("Files generated:")
    print(f"  - {markdown_path}")
    print(f"  - {json_path}")


if __name__ == "__main__":
    main()