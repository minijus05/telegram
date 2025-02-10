import asyncio
import re
from typing import Dict, List
from telethon import TelegramClient, events
import logging

# Konfigūracija
class Config:
    # Telegram settings
    TELEGRAM_API_ID = '25425140'
    TELEGRAM_API_HASH = 'bd0054bc5393af360bc3930a27403c33'
    TELEGRAM_SOURCE_CHATS = ['@solearlytrending', '@botubotass']
    TELEGRAM_DEST_CHAT = '@smartas1'
    
    # Scanner settings
    SCANNER_GROUP = '@skaneriss'
    SOUL_SCANNER_BOT = 6872314605
    SYRAX_SCANNER_BOT = 7488438206
    PROFICY_PRICE_BOT = 5457577145  # Pridėtas ProficyPriceBot ID

# Logging setup
logging.basicConfig(format='[%(levelname) 5s/%(asctime)s] %(name)s: %(message)s',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

config = Config()

# Telegram client setup
client = TelegramClient('scanner', config.TELEGRAM_API_ID, config.TELEGRAM_API_HASH)

# Duomenų saugojimas
scanner_data = {}

# Helper functions
def clean_line(text: str) -> str:
    """
    Išvalo tekstą nuo nereikalingų simbolių, bet palieka svarbius emoji
    """
    important_emoji = ['💠', '🤍', '✅', '❌', '🔻', '🐟', '🍤', '🐳', '🌱', '🕒', '📈', '⚡️', '👥', '🔗', '🦅', '🔫', '⚠️', '🛠', '🔝', '🔥', '💧', '😳', '🤔', '🚩', '📦', '🎯',
        '👍', '💰', '💼']
    
    # 1. Pašalinam Markdown žymėjimą
    cleaned = re.sub(r'\*\*', '', text)
    
    # 2. Pašalinam URL formatu [text](url)
    cleaned = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cleaned)
    
    # 3. Pašalinam likusius URL skliaustus (...)
    cleaned = re.sub(r'\((?:https?:)?//[^)]+\)', '', text)

    # Pašalinam visus specialius simbolius, išskyrus svarbius emoji
    result = ''
    i = 0
    while i < len(cleaned):
        if any(cleaned.startswith(emoji, i) for emoji in important_emoji):
            # Jei randame svarbų emoji, jį paliekame
            emoji_found = next(emoji for emoji in important_emoji if cleaned.startswith(emoji, i))
            result += emoji_found
            i += len(emoji_found)
        else:
            # Kitaip tikriname ar tai normalus simbolis
            if cleaned[i].isalnum() or cleaned[i] in ' .:$%|-()':
                result += cleaned[i]
            i += 1
    
    return result.strip()

def parse_soul_scanner_response(text: str) -> Dict:
    """Parse Soul Scanner message"""
    try:
        data = {}
        lines = text.split('\n')

        for line in lines:
            try:
                if not line.strip():
                    continue
                    
                clean_line_text = clean_line(line)
                
                # Basic info 
                if '💠' in line or '🔥' in line:
                    parts = line.split('$')
                    data['name'] = parts[0].replace('💠', '').replace('🔥', '').replace('•', '').replace('**', '').strip()
                    data['symbol'] = parts[1].replace('**', '').strip()
                        
                # Contract Address
                elif len(line.strip()) > 30 and not any(x in line for x in ['https://', '🌊', '🔫', '📈', '🔗', '•', '┗', '┣']):
                    data['contract_address'] = line.strip().replace('`', '')
                
                                                            
                # Market Cap and ATH
                elif 'MC:' in line:
                    # Market Cap gali būti K arba M
                    mc_k = re.search(r'\$(\d+\.?\d*)K', clean_line_text)  # Ieškome K
                    mc_m = re.search(r'\$(\d+\.?\d*)M', clean_line_text)  # Ieškome M
                    
                    if mc_m:  # Jei M (milijonai)
                        data['market_cap'] = float(mc_m.group(1)) * 1000000
                    elif mc_k:  # Jei K (tūkstančiai)
                        data['market_cap'] = float(mc_k.group(1)) * 1000
                            
                    # ATH ieškojimas (po 🔝)
                    ath_m = re.search(r'🔝 \$(\d+\.?\d*)M', clean_line_text)  # Pirma tikrinam M
                    ath_k = re.search(r'🔝 \$(\d+\.?\d*)K', clean_line_text)  # Tada K
                    
                    if ath_m:  # Jei M (milijonai)
                        data['ath_market_cap'] = float(ath_m.group(1)) * 1000000
                    elif ath_k:  # Jei K (tūkstančiai)
                        data['ath_market_cap'] = float(ath_k.group(1)) * 1000
                
                # Liquidity
                elif 'Liq:' in line:
                    liq = re.search(r'\$(\d+\.?\d*)K\s*\((\d+)\s*SOL\)', clean_line_text)
                    if liq:
                        data['liquidity'] = {
                            'usd': float(liq.group(1)) * 1000,
                            'sol': float(liq.group(2))
                        }
                
                
                
                # Tikriname visą eilutę su Mint ir Freeze
                elif '➕ Mint' in line and '🧊 Freeze' in line:
                    mint_part = line.split('|')[0]
                    freeze_part = line.split('|')[1]
                    data['mint_status'] = False if '🤍' in mint_part else True
                    data['freeze_status'] = False if '🤍' in freeze_part else True

                # LP statusas - GRĮŽTAM PRIE TO KAS VEIKĖ
                elif 'LP' in line and not 'First' in line:
                    data['lp_status'] = True if '🤍' in line else False

                    
                # DEX Status
                elif 'Dex' in line:
                    data['dex_status'] = {
                        'paid': '✅' in line,
                        'ads': not '❌' in line
                    }
                
                # Scans
                elif any(emoji in line for emoji in ['⚡', '⚡️']) and 'Scans:' in line:  # Patikriname abu variantus
                    
                    try:
                        # Pašalinam Markdown formatavimą ir ieškome skaičiaus
                        clean_line_text = re.sub(r'\*\*', '', line)
                        scans_match = re.search(r'Scans:\s*(\d+)', clean_line_text)
                        if scans_match:
                            scan_count = int(scans_match.group(1))
                            data['total_scans'] = scan_count
                            
                            
                        # Social links jau veikia teisingai, paliekame kaip yra
                        social_links = {}
                        if 'X' in line:
                            x_match = re.search(r'X\]\((https://[^)]+)\)', line)
                            if x_match:
                                social_links['X'] = x_match.group(1)

                        if 'TG' in line:
                            tg_match = re.search(r'TG\]\((https://[^)]+)\)', line)
                            if tg_match:
                                social_links['TG'] = tg_match.group(1)
                        
                        if 'WEB' in line:
                            web_match = re.search(r'WEB\]\((https://[^)]+)\)', line)
                            if web_match:
                                social_links['WEB'] = web_match.group(1)
                        
                        if social_links:
                            data['social_links'] = social_links
                            
                    except Exception as e:
                        logger.exception("Scans error:")
                                           
            except Exception as e:
                logger.exception("Error parsing message:")
                return {}
        return data

    except Exception as e:
        logger.exception("Error parsing message:")
        return {}

def parse_syrax_scanner_response(text: str) -> Dict:
    """Parse Syrax Scanner message"""
    try:
        data = {
            'dev_bought': {'tokens': 0.0, 'sol': 0.0, 'percentage': 0.0, 'curve_percentage': 0.0},
            'dev_created_tokens': 0,
            'same_name_count': 0,
            'same_website_count': 0,
            'same_telegram_count': 0,
            'same_twitter_count': 0,
            'bundle': {'count': 0, 'supply_percentage': 0.0, 'curve_percentage': 0.0, 'sol': 0.0},
            'notable_bundle': {'count': 0, 'supply_percentage': 0.0, 'curve_percentage': 0.0, 'sol': 0.0},  # Naujas
            'sniper_activity': {'tokens': 0.0, 'percentage': 0.0, 'sol': 0.0}
        }

        if not text:
            logger.warning(f"Empty text received")
            return data
            
        lines = text.split('\n')
        
        for line in lines:
            try:
                clean_line_text = clean_line(line)
                
                # Dev bought info
                if 'Dev bought' in clean_line_text:
                    tokens_match = re.search(r'(\d+\.?\d*)([KMB]) tokens', clean_line_text)
                    sol_match = re.search(r'(\d+\.?\d*) SOL', clean_line_text)
                    percentage_match = re.search(r'(\d+\.?\d*)%', clean_line_text)
                    curve_match = re.search(r'(\d+\.?\d*)% of curve', clean_line_text)
                    
                    if tokens_match:
                        value = float(tokens_match.group(1))
                        multiplier = {'K': 1000, 'M': 1000000, 'B': 1000000000}[tokens_match.group(2)]
                        data['dev_bought']['tokens'] = value * multiplier
                    if sol_match:
                        data['dev_bought']['sol'] = float(sol_match.group(1))
                    if percentage_match:
                        data['dev_bought']['percentage'] = float(percentage_match.group(1))
                    if curve_match:
                        data['dev_bought']['curve_percentage'] = float(curve_match.group(1))
                
                # Bundle info (🚩 Bundled!)
                if '🚩' in clean_line_text and 'Bundled' in clean_line_text:
                    count_match = re.search(r'(\d+) trades', clean_line_text)
                    supply_match = re.search(r'(\d+\.?\d*)%', clean_line_text)
                    curve_match = re.search(r'\((\d+\.?\d*)% of curve\)', clean_line_text)
                    sol_match = re.search(r'(\d+\.?\d*) SOL', clean_line_text)
                    
                    if count_match:
                        data['bundle']['count'] = int(count_match.group(1))
                    if supply_match:
                        data['bundle']['supply_percentage'] = float(supply_match.group(1))
                    if curve_match:
                        data['bundle']['curve_percentage'] = float(curve_match.group(1))
                    if sol_match:
                        data['bundle']['sol'] = float(sol_match.group(1))
                
                # Notable bundle info (📦 notable bundle(s))
                if '📦' in clean_line_text and 'notable bundle' in clean_line_text:
                    # 1. PIRMA išvalom URL su skliaustais ir kablelius po jų
                    clean_text = re.sub(r'\(http[^)]+\),', '', clean_line_text)

                    # 2. TADA naudojam TUOS PAČIUS regex'us kaip bundle ir sniper
                    count_match = re.search(r'📦\s*(\d+)\s*notable', clean_text)
                    supply_match = re.search(r'(\d+\.?\d*)%\s*of\s*supply', clean_text)
                    curve_match = re.search(r'\((\d+\.?\d*)%\s*of\s*curve\)', clean_text)
                    sol_match = re.search(r'(\d+\.?\d*)\s*SOL', clean_text)
                  
                    if count_match:
                        data['notable_bundle']['count'] = int(count_match.group(1))
                    if supply_match:
                        data['notable_bundle']['supply_percentage'] = float(supply_match.group(1))
                    if curve_match:
                        data['notable_bundle']['curve_percentage'] = float(curve_match.group(1))
                    if sol_match:
                        data['notable_bundle']['sol'] = float(sol_match.group(1))
                        
                # Sniper activity
                if '🎯' in clean_line_text and 'Notable sniper activity' in clean_line_text:
                    tokens_match = re.search(r'(\d+\.?\d*)M', clean_line_text)           # tik M skaičiams
                    percentage_match = re.search(r'\((\d+\.?\d*)%\)', clean_line_text)   # bazinis regex procentams skliausteliuose 
                    sol_match = re.search(r'(\d+\.?\d*) SOL', clean_line_text)          # tas pats formatas visiems
                    
                    if tokens_match:
                        data['sniper_activity']['tokens'] = float(tokens_match.group(1)) * 1000000
                    if percentage_match:
                        data['sniper_activity']['percentage'] = float(percentage_match.group(1))
                    if sol_match:
                        data['sniper_activity']['sol'] = float(sol_match.group(1))
                
                # Dev created tokens
                elif 'Dev created' in clean_line_text:
                    match = re.search(r'Dev created (\d+)', clean_line_text)
                    if match:
                        data['dev_created_tokens'] = int(match.group(1))
                
                # Same name count
                elif 'same as' in clean_line_text and 'name' in clean_line_text.lower():
                    match = re.search(r'same as (\d+)', clean_line_text)
                    if match:
                        data['same_name_count'] = int(match.group(1))
                
                # Same website count
                elif 'same as' in clean_line_text and 'website' in clean_line_text.lower():
                    match = re.search(r'same as (\d+)', clean_line_text)
                    if match:
                        data['same_website_count'] = int(match.group(1))
                
                # Same telegram count
                elif 'same as' in clean_line_text and 'telegram' in clean_line_text.lower():
                    match = re.search(r'same as (\d+)', clean_line_text)
                    if match:
                        data['same_telegram_count'] = int(match.group(1))
                
                # Same twitter count
                elif 'same as' in clean_line_text and 'twitter' in clean_line_text.lower():
                    match = re.search(r'same as (\d+)', clean_line_text)
                    if match:
                        data['same_twitter_count'] = int(match.group(1))
                
            except Exception as e:
                logger.exception(f"Error parsing line '{line}':")
                return {}  # Return an empty dictionary in case of parsing errors

        #logger.info(f"Successfully parsed Syrax data: {data}")
        return data

    except Exception as e:
        logger.exception("Main parsing error:")
        return {}

def parse_proficy_price_bot_response(text: str) -> Dict:
    """Parse ProficyPriceBot message."""
    try:
        data = {}
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("5M:"):
                parts = line.split("   ")
                if len(parts) == 3:
                    data['5M_Price'] = parts[0].split(":")[1].strip()
                    data['5M_Volume'] = parts[1].strip()
                    data['5M_BS'] = parts[2].strip()

            elif line.startswith("1H:"):
                parts = line.split("   ")
                if len(parts) == 3:
                    data['1H_Price'] = parts[0].split(":")[1].strip()
                    data['1H_Volume'] = parts[1].strip()
                    data['1H_BS'] = parts[2].strip()
                    
        return data
    except Exception as e:
        logger.exception("Error parsing ProficyPriceBot message:")
        return {}

def _extract_token_addresses(message: str) -> List[str]:
    """Ištraukia token adresus iš žinutės"""
    matches = []
    
    try:
        logger.debug(f"Extracting token addresses from message: {message}")
        
        # Skaidome žinutę į eilutes
        lines = message.split('\n')
        for line in lines:
            # 1. Originalus formatas (🪙 CA: `token`)
            ca_matches = re.findall(r'🪙\s*CA:\s*`([A-Za-z0-9]+)`', line)
            if ca_matches:
                matches.extend(ca_matches)
                
            # 2. Mint: formatas
            mint_matches = re.findall(r'Mint:\s*([A-Za-z0-9]{32,44})', line)
            if mint_matches:
                matches.extend(mint_matches)
            
            # 3. Tiesioginiai token adresai
            direct_matches = re.findall(r'(?:^|\s)[`"\']?([A-Za-z0-9]{32,44})[`"\']?(?:\s|$)', line)
            if direct_matches:
                matches.extend(direct_matches)
                
            # 4. Token adresai iš URL (tik eilutėse su "New")
            cleaned_line = re.sub(r'[*_~`]', '', line)
            if "New" in cleaned_line:
                url_patterns = [
                    r'birdeye\.so/token/([A-Za-z0-9]{32,44})',
                    r'raydium\.io/swap/\?inputCurrency=([A-Za-z0-9]{32,44})',
                    r'dexscreener\.com/solana/([A-Za-z0-9]{32,44})',
                    r'dextools\.io/app/solana/pair-explorer/([A-Za-z0-9]{32,44})',
                    r'gmgn\.ai/sol/token/([A-Za-z0-9]{32,44})',
                    r'soul_sniper_bot\?start=\d+_([A-Za-z0-9]{32,44})',
                    r'soul_scanner_bot/chart\?startapp=([A-Za-z0-9]{32,44})'
                ]
                
                for pattern in url_patterns:
                    url_matches = re.findall(pattern, line)
                    if url_matches:
                        matches.extend(url_matches)
        
        # Pašaliname dublikatus ir filtruojame
        unique_matches = list(set(matches))
        logger.debug(f"Unique matches: {unique_matches}")
        valid_matches = [addr for addr in unique_matches if len(addr) >= 32 and len(addr) <= 44]
        logger.debug(f"Valid matches: {valid_matches}")
        
        if valid_matches:
            logger.info(f"Found token address: {valid_matches[0]}")
        else:
            logger.warning("No token address found in message")
        
        return valid_matches
        
    except Exception as e:
        logger.exception("Error extracting token address:")
        return []

# Event handler'is žinučių gavimui iš šaltinių kanalų
@client.on(events.NewMessage(chats=config.TELEGRAM_SOURCE_CHATS))
async def handle_new_message(event):
    """Handles messages from source chats, extracts token address and forwards to scanner group."""
    logger.debug("handle_new_message called")
    try:
        logger.debug(f"Message from chat ID: {event.chat_id}")
        logger.debug(f"Message content: {event.message.message}")

        message_text = event.message.message
        token_addresses = _extract_token_addresses(message_text)
        if token_addresses:
            token_address = token_addresses[0]  # Take the first address if multiple are found
            logger.info(f"Extracted token address: {token_address}")

            # Send the token address to the scanner group
            try:
                await client.send_message(config.SCANNER_GROUP, token_address)
                logger.info(f"Sent token address to scanner group: {config.SCANNER_GROUP}")

                # Initialize data storage for this token address
                scanner_data[token_address] = {
                    "soul": None,
                    "syrax": None,
                    "proficy": None
                }

            except Exception as e:
                logger.exception(f"Error sending message to scanner group: {config.SCANNER_GROUP}")

        else:
            logger.debug("No token address found in message")

    except Exception as e:
        logger.exception("Error handling new message:")

# Event handler'iai žinučių gavimui iš scanner group
@client.on(events.NewMessage(chats=config.SCANNER_GROUP, from_users=[config.SOUL_SCANNER_BOT]))
async def handle_soul_scanner(event):
    """Handles messages from Soul Scanner bot."""
    await process_scanner_message(event, "soul")

@client.on(events.NewMessage(chats=config.SCANNER_GROUP, from_users=[config.SYRAX_SCANNER_BOT]))
async def handle_syrax_scanner(event):
    """Handles messages from Syrax Scanner bot."""
    await process_scanner_message(event, "syrax")

@client.on(events.NewMessage(chats=config.SCANNER_GROUP, from_users=[config.PROFICY_PRICE_BOT]))
async def handle_proficy_price_bot(event):
    """Handles messages from ProficyPriceBot."""
    await process_scanner_message(event, "proficy")

async def process_scanner_message(event, scanner_type):
    """Processes scanner messages and prints the combined output."""
    logger.debug(f"Processing {scanner_type} scanner message")
    try:
        message_text = event.message.message
        # Extract token address from the message (assuming it's the first line)
        token_address = message_text.split('\n')[0].strip()

        # Check if this token address is being tracked
        if token_address in scanner_data:
            if scanner_type == "soul":
                scanner_data[token_address]["soul"] = parse_soul_scanner_response(message_text)
            elif scanner_type == "syrax":
                scanner_data[token_address]["syrax"] = parse_syrax_scanner_response(message_text)
            elif scanner_type == "proficy":
                scanner_data[token_address]["proficy"] = parse_proficy_price_bot_response(message_text)
            
            logger.info(f"Received {scanner_type} data for token: {token_address}")
            await print_combined_data(token_address)  # Print combined data if all scanners responded

    except Exception as e:
        logger.exception(f"Error handling {scanner_type} message:")

async def print_combined_data(token_address):
    """Prints the combined scanner data in a formatted way."""
    if (scanner_data[token_address]["soul"] and
            scanner_data[token_address]["syrax"] and
            scanner_data[token_address]["proficy"]):

        print(f"\nNaujas token adresas: {token_address}\n")

        soul_data = scanner_data[token_address]["soul"]
        print("Soul Scanner Data:")
        print(f"  Name: {soul_data.get('name', 'N/A')}")
        print(f"  Symbol: {soul_data.get('symbol', 'N/A')}")
        print(f"  Contract Address: {soul_data.get('contract_address', 'N/A')}")
        print(f"  Market Cap: {soul_data.get('market_cap', 'N/A')}")
        print(f"  ATH Market Cap: {soul_data.get('ath_market_cap', 'N/A')}")
        if soul_data.get('liquidity'):
            print(f"  Liquidity USD: {soul_data['liquidity'].get('usd', 'N/A')}")
            print(f"  Liquidity SOL: {soul_data['liquidity'].get('sol', 'N/A')}")
        else:
            print("  Liquidity: N/A")
        print(f"  Mint Status: {soul_data.get('mint_status', 'N/A')}")
        print(f"  Freeze Status: {soul_data.get('freeze_status', 'N/A')}")
        if soul_data.get('dex_status'):
            print(f"  Dex Status Paid: {soul_data['dex_status'].get('paid', 'N/A')}")
            print(f"  Dex Status Ads: {soul_data['dex_status'].get('ads', 'N/A')}")
        else:
            print("  Dex Status: N/A")
        print(f"  Total Scans: {soul_data.get('total_scans', 'N/A')}")

        if soul_data.get('social_links'):
            social_links = soul_data['social_links']
            if social_links.get('X'):
                print(f"  Social Links X: {social_links['X']}")
            if social_links.get('TG'):
                print(f"  Social Links TG: {social_links['TG']}")
            if social_links.get('WEB'):
                print(f"  Social Links WEB: {social_links['WEB']}")

        syrax_data = scanner_data[token_address]["syrax"]
        print("\nSyrax Scanner Data:")
        print(f"  Dev Bought Tokens: {syrax_data['dev_bought'].get('tokens', 'N/A')}")
        print(f"  Dev Bought SOL: {syrax_data['dev_bought'].get('sol', 'N/A')}")
        print(f"  Dev Bought Percentage: {syrax_data['dev_bought'].get('percentage', 'N/A')}")
        print(f"  Dev Bought Curve Percentage: {syrax_data['dev_bought'].get('curve_percentage', 'N/A')}")
        print(f"  Dev Created Tokens: {syrax_data.get('dev_created_tokens', 'N/A')}")
        print(f"  Same Name Count: {syrax_data.get('same_name_count', 'N/A')}")
        print(f"  Same Website Count: {syrax_data.get('same_website_count', 'N/A')}")
        print(f"  Same Telegram Count: {syrax_data.get('same_telegram_count', 'N/A')}")
        print(f"  Same Twitter Count: {syrax_data.get('same_twitter_count', 'N/A')}")
        print(f"  Bundle Count: {syrax_data['bundle'].get('count', 'N/A')}")
        print(f"  Bundle Supply Percentage: {syrax_data['bundle'].get('supply_percentage', 'N/A')}")
        print(f"  Bundle Curve Percentage: {syrax_data['bundle'].get('curve_percentage', 'N/A')}")
        print(f"  Bundle SOL: {syrax_data['bundle'].get('sol', 'N/A')}")
        print(f"  Notable Bundle Count: {syrax_data['notable_bundle'].get('count', 'N/A')}")
        print(f"  Notable Bundle Supply Percentage: {syrax_data['notable_bundle'].get('supply_percentage', 'N/A')}")
        print(f"  Notable Bundle Curve Percentage: {syrax_data['notable_bundle'].get('curve_percentage', 'N/A')}")
        print(f"  Notable Bundle SOL: {syrax_data['notable_bundle'].get('sol', 'N/A')}")
        print(f"  Sniper Activity Tokens: {syrax_data['sniper_activity'].get('tokens', 'N/A')}")
        print(f"  Sniper Activity Percentage: {syrax_data['sniper_activity'].get('percentage', 'N/A')}")
        print(f"  Sniper Activity SOL: {syrax_data['sniper_activity'].get('sol', 'N/A')}")

        proficy_data = scanner_data[token_address]["proficy"]
        print("\nProficy Price Bot Data:")
        print(f"  5M Price: {proficy_data.get('5M_Price', 'N/A')}")
        print(f"  5M Volume: {proficy_data.get('5M_Volume', 'N/A')}")
        print(f"  5M B/S: {proficy_data.get('5M_BS', 'N/A')}")
        print(f"  1H Price: {proficy_data.get('1H_Price', 'N/A')}")
        print(f"  1H Volume: {proficy_data.get('1H_Volume', 'N/A')}")
        print(f"  1H B/S: {proficy_data.get('1H_BS', 'N/A')}")

        # Remove data for this token address
        del scanner_data[token_address]
    else:
        logger.warning(f"Not all scanner data received for {token_address} yet.")

async def main():
    try:
        await client.connect()
        await client.start()
        print("Telegram scanner bot started...")
        await client.run_until_disconnected()
    except Exception as e:
        logger.exception("Error during startup:")

if __name__ == '__main__':
    asyncio.run(main())
