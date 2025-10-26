#!/usr/bin/env python3

# Test script to debug import issues

try:
    print("Importing trim module...")
    import fit.trim as trim_module
    print("Trim module imported successfully")
    
    print("Available functions in trim module:")
    functions = [name for name in dir(trim_module) if not name.startswith('_') and callable(getattr(trim_module, name))]
    for func in functions:
        print(f"  - {func}")
    
    print(f"\nChecking if 'remove_redundant_point' is available:")
    if hasattr(trim_module, 'remove_redundant_point'):
        print("✓ remove_redundant_point is available")
    else:
        print("✗ remove_redundant_point is NOT available")
        
    print("\nTrying to import specific function:")
    from fit.trim import remove_redundant_point
    print("✓ Successfully imported remove_redundant_point")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()