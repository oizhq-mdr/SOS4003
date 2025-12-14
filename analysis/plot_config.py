# """
# Matplotlib configuration for Korean text support with Inter font
# """
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# from matplotlib import rcParams

# def setup_korean_fonts():
#     """
#     Setup matplotlib to display Korean text properly
#     Uses Inter for English and AppleGothic/Arial Unicode MS for Korean
#     """
#     # Try to use Inter for English if available, otherwise use default sans-serif
#     try:
#         # Set font family with fallback
#         rcParams['font.family'] = 'sans-serif'
#         rcParams['font.sans-serif'] = ['Inter', 'AppleGothic', 'Arial Unicode MS', 'DejaVu Sans']
#     except:
#         # Fallback to AppleGothic if Inter is not available
#         rcParams['font.family'] = 'sans-serif'
#         rcParams['font.sans-serif'] = ['AppleGothic', 'Arial Unicode MS', 'DejaVu Sans']
    
#     # Ensure minus sign displays correctly
#     rcParams['axes.unicode_minus'] = False
    
#     # Set default figure DPI for better quality
#     rcParams['figure.dpi'] = 100
#     rcParams['savefig.dpi'] = 300
    
#     # Set default figure size
#     rcParams['figure.figsize'] = (12, 6)
    
#     # Improve text rendering
#     rcParams['text.antialiased'] = True 
    
#     print("✓ Korean font support configured")
#     print(f"  Font family: {rcParams['font.sans-serif']}")

# def get_korean_font():
#     """
#     Get the name of an available Korean font
#     Returns the first available Korean font from the preference list
#     """
#     korean_fonts = ['AppleGothic', 'Arial Unicode MS', 'NanumGothic', 'Malgun Gothic']
    
#     available_fonts = [f.name for f in fm.fontManager.ttflist]
    
#     for font in korean_fonts:
#         if font in available_fonts:
#             return font
    
#     # Fallback
#     return 'AppleGothic'

# # Auto-setup when imported
# setup_korean_fonts()
