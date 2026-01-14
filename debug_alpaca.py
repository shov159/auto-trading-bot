try:
    import alpaca_trade_api.common
    print("alpaca_trade_api.common imported")
    print(dir(alpaca_trade_api.common))
except ImportError as e:
    print(e)

try:
    from alpaca_trade_api.common import URL
    print("URL found in common")
except ImportError:
    print("URL NOT found in common")
    # Try finding it elsewhere
    import alpaca_trade_api
    print("Searching in alpaca_trade_api...")
    # Recursive search or just common places
    try:
        from alpaca_trade_api.rest import URL
        print("Found URL in rest")
    except:
        print("Not in rest")


