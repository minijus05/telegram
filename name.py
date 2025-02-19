import re
import logging

import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from typing import Dict, List
from telethon import TelegramClient, events
import asyncio
from datetime import datetime, timezone, timedelta
import sqlite3
import time
from telethon.sessions.sqlite import SQLiteSession
from contextlib import contextmanager
import os

# Configure logging
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



class Config:
    # Telegram settings
    TELEGRAM_API_ID = '25425140'
    TELEGRAM_API_HASH = 'bd0054bc5393af360bc3930a27403c33'
    TELEGRAM_SOURCE_CHATS = ['@PumpFunRaydium', '@botubotass'] #'@solearlytrending', '@HighVolumeBordga', '@solanahypee'
    
    TELEGRAM_GEM_CHAT = '@testasmano'
    
    # Scanner settings
    SCANNER_GROUP = '@skaneriss'
    SOUL_SCANNER_BOT = 6872314605
    SYRAX_SCANNER_BOT = 7488438206
    PROFICY_PRICE_BOT = 5457577145

    USER_LOGIN = 'minijus05'

    # IŠTRINTI TOKEN IŠ DUOMENŲ BAZĖZ, IŠ VISŲ IŠTRINA
    # I GRUPE, TARKIM @BOTUBOTASS PARASYTI ŽINUTĘ         /delete TOKEN_ADRESAS

    # ML settings
    MIN_GEMS_FOR_ANALYSIS = 5  # Minimalus GEM skaičius prieš pradedant analizę

    # GEM settings
    GEM_MULTIPLIER = "10x"
    MIN_GEM_SCORE = 10

    # Žinutes siuntimas
    MIN_SIMILARITY_SCORE = 1.0
    MIN_CONFIDENCE_LEVEL = 0.0
    

class TokenMonitor:
    def __init__(self, monitor_session=None, scanner_session=None):
        if isinstance(monitor_session, SQLiteSession):
            self.telegram = TelegramClient(monitor_session, 
                                       Config.TELEGRAM_API_ID, 
                                       Config.TELEGRAM_API_HASH)
        else:
            self.telegram = TelegramClient('token_monitor_session', 
                                       Config.TELEGRAM_API_ID, 
                                       Config.TELEGRAM_API_HASH)
        
        if isinstance(scanner_session, SQLiteSession):
            self.scanner_client = TelegramClient(scanner_session,
                                             Config.TELEGRAM_API_ID,
                                             Config.TELEGRAM_API_HASH)
        else:
            self.scanner_client = TelegramClient('scanner_session',
                                             Config.TELEGRAM_API_ID,
                                             Config.TELEGRAM_API_HASH)
            
        self.db = DatabaseManager()
        self.gem_analyzer = MLGEMAnalyzer()
        self.logger = logger
        
        # Įrašome bot'o paleidimo informaciją
        self.db.cursor.execute('''
        INSERT INTO bot_info (start_time, user_login, last_active)
        VALUES (?, ?, ?)
        ''', (
            datetime.now(timezone.utc),
            "minijus05",
            datetime.now(timezone.utc)
        ))
        self.db.conn.commit()

    
    async def initialize(self):
        """Initialize clients"""
        await self.telegram.start()
        await self.scanner_client.start()
        
        # Atnaujiname last_active laiką
        self.db.cursor.execute('''
        UPDATE bot_info 
        SET last_active = ? 
        WHERE user_login = ?
        ''', (datetime.now(timezone.utc), "minijus05"))
        self.db.conn.commit()
        
        return self

    async def handle_new_message(self, event):
        try:
            message = event.message.text
            token_addresses = []
            
            # Tikriname ar žinutė turi reply_to (atsakymas į kitą žinutę)
            if event.message.reply_to:
                try:
                    # Gauname originalią žinutę į kurią atsakyta
                    replied_msg = await event.message.get_reply_message()
                    if replied_msg and replied_msg.text:
                        reply_addresses = self._extract_token_addresses(replied_msg.text)
                        token_addresses.extend(reply_addresses)
                except Exception as e:
                    logger.error(f"Error getting reply message: {e}")

            # Tikriname pagrindinę žinutę
            main_addresses = self._extract_token_addresses(message)
            token_addresses.extend(main_addresses)
            
            # Pašaliname dublikatus
            token_addresses = list(set(token_addresses))
                        
            
            if token_addresses:
                for address in token_addresses:
                    is_new_token = "New" in message.lower() or "Marketcap" or "migration" in message
                    is_from_token = "from" in message.lower() or "MADE" in message or "🔝" in message
                    
                    # Patikriname ar token'as jau yra DB
                    self.db.cursor.execute("SELECT address FROM tokens WHERE address = ?", (address,))
                    token_exists = self.db.cursor.fetchone() is not None
                    
                    if is_new_token:
                        # Jei token'as jau egzistuoja - praleidžiam
                        if token_exists:
                            print(f"\n[SKIPPED NEW] Token already exists in database: {address}")
                            continue
                            
                        print(f"\n[NEW TOKEN DETECTED] Address: {address}")
                        # Siunčiame į scanner grupę
                        original_message = await self.scanner_client.send_message(
                            Config.SCANNER_GROUP,
                            address
                        )
                        logger.info(f"Sent NEW token to scanner group: {address}")
                        
                        # Renkame scannerių duomenis
                        scanner_data = await self._collect_scanner_data(address, original_message)
                        
                        if scanner_data:
                            # Išsaugome token duomenis į DB
                            self.db.save_token_data(
                                address,
                                scanner_data['soul'],
                                scanner_data['syrax'],
                                scanner_data['proficy'],
                                is_new_token=True
                            )
                            print(f"[SUCCESS] Saved NEW token data: {address}")

                            # ANALIZUOJAME TIK NAUJUS TOKENUS
                            analysis_result = self.gem_analyzer.analyze_token(scanner_data)

                            if analysis_result['status'] == 'pending':
                                print(f"\n[ANALYSIS PENDING] {analysis_result['message']}")
                            else:
                                await self._handle_analysis_results(analysis_result, scanner_data)
                            
                    elif is_from_token:
                        if not token_exists:
                            print(f"\n[SKIPPED UPDATE] Token not found in database: {address}")
                            continue
                            
                        print(f"\n[UPDATE TOKEN DETECTED] Address: {address}")
                        # Siunčiame į scanner grupę
                        original_message = await self.scanner_client.send_message(
                            Config.SCANNER_GROUP,
                            address
                        )
                        logger.info(f"Sent token UPDATE to scanner group: {address}")
                        
                        # Renkame scannerių duomenis
                        scanner_data = await self._collect_scanner_data(address, original_message)
                        
                        if scanner_data:
                            # Atnaujiname token duomenis DB
                            self.db.save_token_data(
                                address,
                                scanner_data['soul'],
                                scanner_data['syrax'],
                                scanner_data['proficy'],
                                is_new_token=False
                            )
                            print(f"[SUCCESS] Updated existing token data: {address}")
                            
                            # Jei tai "from" žinutė su "10x" - pridedame į GEM duomenų bazę
                            if Config.GEM_MULTIPLIER in event.message.text:
                                self.gem_analyzer.add_gem_token(scanner_data)
                                print(f"[GEM ADDED] Token marked as GEM: {address}")
                            
                            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            print(f"[ERROR] Message handling failed: {e}")

    async def handle_delete_command(self, event):
        """Handles the /delete command to remove tokens from database"""
        try:
            # Gauname komandos tekstą
            message = event.message.text
            
            # Patikriname ar yra tokeno adresas
            parts = message.split()
            if len(parts) != 2:
                await event.reply("❌ Please use format: /delete TOKEN_ADDRESS")
                return
                
            token_address = parts[1].strip()
            
            # Patikriname ar token'as egzistuoja
            self.db.cursor.execute("SELECT address FROM tokens WHERE address = ?", (token_address,))
            if not self.db.cursor.fetchone():
                await event.reply(f"❌ Token {token_address} not found in database")
                return
                
            try:
                # Ištriname susijusius duomenis iš visų lentelių
                self.db.cursor.execute('BEGIN TRANSACTION')
                
                # Triname duomenis iš visų lentelių pagal eiliškumą (dėl foreign key constraints)
                tables = [
                    'token_analysis_results',
                    'proficy_price_data',
                    'syrax_scanner_data',
                    'soul_scanner_data',
                    'gem_tokens',
                    'tokens'
                ]
                
                for table in tables:
                    if table == 'tokens':
                        self.db.cursor.execute(f"DELETE FROM {table} WHERE address = ?", (token_address,))
                    else:
                        self.db.cursor.execute(f"DELETE FROM {table} WHERE token_address = ?", (token_address,))
                
                self.db.conn.commit()  # Naudojame conn vietoj connection
                await event.reply(f"✅ Successfully deleted token {token_address} and all related data")
                
            except Exception as e:
                self.db.conn.rollback()  # Naudojame conn vietoj connection
                logger.error(f"Database error while deleting token: {e}")
                await event.reply("❌ Database error occurred while deleting token")
                
        except Exception as e:
            logger.error(f"Error handling delete command: {e}")
            await event.reply("❌ Error occurred while deleting token")

    async def _collect_scanner_data(self, address, original_message):
        """Renka duomenis iš visų scannerių"""
        timeout = 15
        start_time = time.time()
        last_check_time = 0
        scanner_data = {
            "soul": None,
            "syrax": None,
            "proficy": None
        }

        while time.time() - start_time < timeout:
            if time.time() - last_check_time >= 1:
                last_check_time = time.time()

                async for message in self.scanner_client.iter_messages(
                    Config.SCANNER_GROUP,
                    limit=10,
                    min_id=original_message.id
                ):
                    if message.sender_id == Config.SOUL_SCANNER_BOT:
                        scanner_data["soul"] = self.parse_soul_scanner_response(message.text)
                    elif message.sender_id == Config.SYRAX_SCANNER_BOT:
                        scanner_data["syrax"] = self.parse_syrax_scanner_response(message.text)
                    elif message.sender_id == Config.PROFICY_PRICE_BOT:
                        scanner_data["proficy"] = await self.parse_proficy_price(message.text)

                    if all(scanner_data.values()):
                        return scanner_data

            await asyncio.sleep(1)
        
        return None

    async def _handle_analysis_results(self, analysis_result, scanner_data):
        """Formatuoja ir rodo analizės rezultatus"""
        print("\n" + "="*50)
        print(f"ML ANALYSIS RESULTS AT {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("="*50)
        
        if analysis_result['status'] == 'pending':
            print(f"\n[ANALYSIS PENDING]")
            print(f"Reason: {analysis_result['message']}")
            print(f"Collected GEMs: {analysis_result.get('collected_gems', 0)}")
            
        elif analysis_result['status'] == 'success':
            soul_data = scanner_data.get('soul', {})
            print(f"\nAnalyzing Token: {soul_data.get('name', 'Unknown')} (${soul_data.get('symbol', 'Unknown')})")
            
            print("\n--- PRIMARY PARAMETERS CHECK ---")
            syrax_data = scanner_data.get('syrax', {})
            
            # Syrax Scanner Parametrai
            print("\nSyrax Scanner Parameters:")
            print(f"Dev Created Tokens: {syrax_data.get('dev_created_tokens', 0)}")
            print(f"Similar Tokens:")
            print(f"- Same Name: {syrax_data.get('same_name_count', 0)}")
            print(f"- Same Website: {syrax_data.get('same_website_count', 0)}")
            print(f"- Same Telegram: {syrax_data.get('same_telegram_count', 0)}")
            print(f"- Same Twitter: {syrax_data.get('same_twitter_count', 0)}")
            print(f"Dev Activity:")

            # Dev bought info - nested structure
            dev_bought = syrax_data.get('dev_bought', {})
            print(f"- Bought %: {dev_bought.get('percentage', 0)}")
            print(f"- Bought Curve %: {dev_bought.get('curve_percentage', 0)}")

            # Dev sold info - nested structure
            dev_sold = syrax_data.get('dev_sold', {})
            print(f"- Sold %: {dev_sold.get('percentage', 0)}")

            # Holders info - nested structure
            holders = syrax_data.get('holders', {})
            print(f"Holders Distribution:")
            print(f"- Total Holders: {holders.get('total', 0)}")
            print(f"- Top 10% Hold: {holders.get('top10_percentage', 0)}%")
            print(f"- Top 25% Hold: {holders.get('top25_percentage', 0)}%")
            print(f"- Top 50% Hold: {holders.get('top50_percentage', 0)}%")

            # Soul Scanner Parametrai
            print("\nSoul Scanner Parameters:")
            print(f"Market Cap: ${soul_data.get('market_cap', 0):,.2f}")
            print(f"Liquidity USD: ${soul_data.get('liquidity_usd', 0):,.2f}")

            # Proficy Parametrai - nested structure
            proficy_data = scanner_data.get('proficy', {})
            hour_data = proficy_data.get('1h', {})
            print("\nProficy Parameters:")
            print(f"1h Volume: ${hour_data.get('volume', 0):,.2f}")
            print(f"1h Price Change: {hour_data.get('price_change', 0)}%")
            print(f"1h B/S Ratio: {hour_data.get('bs_ratio', '1/1')}")
            
            print("\n--- PARAMETER RANGES CHECK ---")
            for param, details in analysis_result['primary_check']['details'].items():
                status = "✅" if details['in_range'] else "❌"
                print(f"\n{status} {param}:")
                print(f"    Current Value: {details['value']:.2f}")
                print(f"    Valid Range: {details['interval']['min']:.2f} - {details['interval']['max']:.2f}")
                print(f"    Z-Score: {details['z_score']:.2f}")
            
            print("\n--- ANALYSIS RESULTS ---")
            print(f"GEM Potential Score: {analysis_result['similarity_score']:.1f}%")
            print(f"Confidence Level: {analysis_result['confidence_level']:.1f}%")
            print(f"Recommendation: {analysis_result['recommendation']}")
            
            # Čia įdedame naują kodą
            if (analysis_result['similarity_score'] >= Config.MIN_SIMILARITY_SCORE and 
                analysis_result['confidence_level'] >= Config.MIN_CONFIDENCE_LEVEL):
                print(f"\n🚀 HIGH GEM POTENTIAL DETECTED!")
                print(f"Similarity Score: {analysis_result['similarity_score']:.1f}% (>= {Config.MIN_SIMILARITY_SCORE}%)")
                print(f"Confidence Level: {analysis_result['confidence_level']:.1f}% (>= {Config.MIN_CONFIDENCE_LEVEL}%)")
                print(f"Sending alert to {Config.TELEGRAM_GEM_CHAT}")
                await self.send_analysis_alert(analysis_result, scanner_data)
            else:
                print(f"\n⚠️ Token does not meet criteria:")
                print(f"Similarity Score: {analysis_result['similarity_score']:.1f}% (need >= {Config.MIN_SIMILARITY_SCORE}%)")
                print(f"Confidence Level: {analysis_result['confidence_level']:.1f}% (need >= {Config.MIN_CONFIDENCE_LEVEL}%)")
                print("No alert sent")
        
        else:  # status == 'failed'
            print("\n[ANALYSIS FAILED]")
            print(f"Stage: {analysis_result['stage']}")
            print(f"Score: {analysis_result['score']}")
            print(f"Message: {analysis_result['message']}")
            
            # Pašalintas dubliuotas Failed Parameters rodymas
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50 + "\n")

    def should_send_alert(self, similarity_score: float, confidence_level: float) -> bool:
        """Tikrina ar reikia siųsti įspėjimą"""
        return (similarity_score >= Config.MIN_SIMILARITY_SCORE and 
                confidence_level >= Config.MIN_CONFIDENCE_LEVEL)

    async def send_analysis_alert(self, analysis_result: Dict, scanner_data: Dict):
        """Siunčia analizės rezultatų žinutę į Telegram"""
        try:
            if not self.should_send_alert(
                analysis_result['similarity_score'], 
                analysis_result['confidence_level']
            ):
                return
            
            message = await self.format_analysis_message(analysis_result, scanner_data)
            
            await self.telegram.send_message(
                Config.TELEGRAM_GEM_CHAT,
                message,
                parse_mode='Markdown',
                link_preview=False
            )
            logger.info(f"Sent analysis alert with {analysis_result['similarity_score']}% similarity")
        except Exception as e:
            logger.error(f"Error sending analysis alert: {e}")
            print(f"[ERROR] Failed to send analysis alert: {str(e)}")

    def get_current_time(self) -> tuple:
        """Grąžina dabartinį UTC ir UTC+2 laiką"""
        utc_now = datetime.now(timezone.utc)
        local_time = utc_now + timedelta(hours=2)
        return (
            utc_now.strftime('%Y-%m-%d %H:%M:%S'),
            local_time.strftime('%Y-%m-%d %H:%M:%S')
        )

    async def format_analysis_message(self, analysis_result: Dict, scanner_data: Dict) -> str:
        """Formatuoja analizės rezultatų žinutę"""
        utc_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        
        soul_data = scanner_data['soul']
        syrax_data = scanner_data['syrax']
        proficy_data = scanner_data['proficy']

        token_address = (
            soul_data.get('token_address') or 
            soul_data.get('address') or 
            soul_data.get('contract_address')
        )

        # Pradžios tekstas
        message = f"""Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {utc_time}
    Current User's Login: minijus05

    --- PARAMETER RANGES CHECK ---"""

        # Parameter ranges check dalis
        for param, details in analysis_result['primary_check']['details'].items():
            status = "✅" if details['in_range'] else "❌"
            message += f"""

    {status} {param}:
        Current Value: {details['value']:.2f}
        Valid Range: {details['interval']['min']:.2f} - {details['interval']['max']:.2f}
        Z-Score: {details['z_score']:.2f}"""

        # Analysis Results dalis
        message += f"""

    --- ANALYSIS RESULTS ---
    GEM Potential Score: {analysis_result['similarity_score']:.1f}%
    Confidence Level: {analysis_result['confidence_level']:.1f}%
    Recommendation: {analysis_result['recommendation']}

    
    Token Address:
    `{token_address}`
    [View on GMGN](https://gmgn.ai/sol/token/{soul_data['contract_address']})

    
    TokenAnalysis"""

        return message

    
    def _extract_token_addresses(self, message: str) -> List[str]:
        """Ištraukia token adresus iš žinutės"""
        try:
            # Pirmiausiai ieškome tiesiogiai pateikto CA (Contract Address)
            ca_match = re.search(r'(?:📃\s*CA:|CA:|🔸\s*Contract Address:\s*)([A-Za-z0-9]{32,44})|(?:\[Token Details\].*?🔸\s*Contract Address:\s*)([A-Za-z0-9]{32,44})', message, re.MULTILINE | re.IGNORECASE | re.DOTALL)
            if ca_match:
                addr = ca_match.group(1) if ca_match.group(1) else ca_match.group(2)
                if 32 <= len(addr) <= 44:
                    logger.info(f"Found token address from CA: {addr}")
                    return [addr]
            
            # Jei CA nerastas, ieškome per URL patterns prioriteto tvarka
            patterns = [
                # Soul Sniper bot (nauji tokenai)
                r'soul_sniper_bot\?start=\d+_([A-Za-z0-9]{32,44})',
                # Soul Scanner chart
                r'soul_scanner_bot/chart\?startapp=([A-Za-z0-9]{32,44})',
                # Soul Scanner start
                r'soul_scanner_bot\?start=([A-Za-z0-9]{32,44})',
                # Rugcheck
                r'rugcheck\.xyz/tokens/([A-Za-z0-9]{32,44})'
            ]
            
            # Ieškome per kiekvieną pattern, kol randame pirmą tinkamą adresą
            for pattern in patterns:
                match = re.search(pattern, message)
                if match:
                    addr = match.group(1)
                    if 32 <= len(addr) <= 44:
                        logger.info(f"Found token address: {addr}")
                        return [addr]
            
            # Jei nieko neradome
            return []
                
        except Exception as e:
            logger.error(f"Error extracting token address: {e}")
            return []
        
    def clean_line(self, text: str) -> str:
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
        cleaned = re.sub(r'\((?:https?:)?//[^)]+\)', '', cleaned)
    
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

    def parse_soul_scanner_response(self, text: str) -> Dict:
        """Parse Soul Scanner message"""
        try:
            data = {}
            lines = text.split('\n')
            
            for line in lines:
                try:
                    if not line.strip():
                        continue
                        
                    clean_line = self.clean_line(line)
                    
                    # Basic info 
                    if '💠' in line or '🔥' in line:
                        parts = line.split('$')
                        data['name'] = parts[0].replace('💠', '').replace('🔥', '').replace('•', '').replace('**', '').strip()
                        data['symbol'] = parts[1].replace('**', '').strip()
                            
                    # Contract Address
                    elif len(line.strip()) > 30 and not any(x in line for x in ['https://', '🌊', '🔫', '📈', '🔗', '•', '┗', '┣', '⚠️', '🚨']):
                        data['contract_address'] = line.strip().replace('`', '')
                    
                                                               
                    # Market Cap and ATH
                    elif 'MC:' in line:
                        # Market Cap gali būti K arba M
                        mc_k = re.search(r'MC: \$(\d+\.?\d*)K', clean_line)  # Ieškome K
                        mc_m = re.search(r'MC: \$(\d+\.?\d*)M', clean_line)  # Ieškome M
                        
                        if mc_m:  # Jei M (milijonai)
                            data['market_cap'] = float(mc_m.group(1)) * 1000000
                        elif mc_k:  # Jei K (tūkstančiai)
                            data['market_cap'] = float(mc_k.group(1)) * 1000
                                
                        # ATH ieškojimas (po 🔝)
                        ath_m = re.search(r'🔝 \$(\d+\.?\d*)M', clean_line)  # Pirma tikrinam M
                        ath_k = re.search(r'🔝 \$(\d+\.?\d*)K', clean_line)  # Tada K
                        
                        if ath_m:  # Jei M (milijonai)
                            data['ath_market_cap'] = float(ath_m.group(1)) * 1000000
                        elif ath_k:  # Jei K (tūkstančiai)
                            data['ath_market_cap'] = float(ath_k.group(1)) * 1000
                    
                    # Liquidity
                    elif 'Liq:' in line:
                        liq = re.search(r'\$(\d+\.?\d*)K\s*\((\d+)\s*SOL\)', clean_line)
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
                    elif any(emoji in line for emoji in ['⚡', '⚡️']) and 'Scans:' in line:
                        try:
                            # Pašalinam Markdown formatavimą ir ieškome skaičiaus
                            clean_line = re.sub(r'\*\*', '', line)
                            scans_match = re.search(r'Scans:\s*(\d+)', clean_line)
                            if scans_match:
                                scan_count = int(scans_match.group(1))
                                data['total_scans'] = scan_count
                                
                            # Social links
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
                            print(f"Scans error: {str(e)}")
                            
                except Exception as e:
                    logger.warning(f"Error parsing line: {str(e)}")
                    continue
                    
            return data

        except Exception as e:
            self.logger.error(f"Error parsing message: {str(e)}")
            return {}

    def parse_syrax_scanner_response(self, text: str) -> Dict:
        """Parse Syrax Scanner message"""
        try:
            # Patikriname ar gavome klaidos pranešimą
            if "🤔 Hmm, I could not scan this token" in text:
                logger.warning("Syrax Scanner could not scan the token")
                return {
                    'error': "Token scan failed - only pump.fun tokens are currently supported",
                    'dev_bought': {'tokens': 'N/A', 'sol': 'N/A', 'percentage': 'N/A', 'curve_percentage': 'N/A'},
                    'dev_created_tokens': 'N/A',
                    'same_name_count': 'N/A',
                    'same_website_count': 'N/A',
                    'same_telegram_count': 'N/A',
                    'same_twitter_count': 'N/A',
                    'bundle': {'count': 'N/A', 'supply_percentage': 'N/A', 'curve_percentage': 'N/A', 'sol': 'N/A'},
                    'notable_bundle': {'count': 'N/A', 'supply_percentage': 'N/A', 'curve_percentage': 'N/A', 'sol': 'N/A'},
                    'sniper_activity': {'tokens': 'N/A', 'percentage': 'N/A', 'sol': 'N/A'},
                    # Nauji laukai
                    'created_time': 'N/A',
                    'traders': {'count': 'N/A', 'last_swap': 'N/A'},
                    'holders': {
                        'total': 'N/A',
                        'top10_percentage': 'N/A',
                        'top25_percentage': 'N/A',
                        'top50_percentage': 'N/A'
                    },
                    'dev_holds': 'N/A',
                    'dev_sold': {'times': 'N/A', 'sol': 'N/A', 'percentage': 'N/A'}
                }

            data = {
                'dev_bought': {'tokens': 0.0, 'sol': 0.0, 'percentage': 0.0, 'curve_percentage': 0.0},
                'dev_created_tokens': 0,
                'same_name_count': 0,
                'same_website_count': 0,
                'same_telegram_count': 0,
                'same_twitter_count': 0,
                'bundle': {'count': 0, 'supply_percentage': 0.0, 'curve_percentage': 0.0, 'sol': 0.0},
                'notable_bundle': {'count': 0, 'supply_percentage': 0.0, 'curve_percentage': 0.0, 'sol': 0.0},
                'sniper_activity': {'tokens': 0.0, 'percentage': 0.0, 'sol': 0.0},
                # Nauji laukai
                'created_time': '',
                'traders': {'count': 0, 'last_swap': ''},
                'holders': {
                    'total': 0,
                    'top10_percentage': 0.0,
                    'top25_percentage': 0.0,
                    'top50_percentage': 0.0
                },
                'dev_holds': 0,
                'dev_sold': {'times': 0, 'sol': 0.0, 'percentage': 0.0}
            }

            lines = text.split('\n')
            
            for line in lines:
                try:
                    clean_line = self.clean_line(line)

                    # Created Time
                    if 'Created:' in clean_line:
                        data['created_time'] = clean_line.split('Created:')[1].strip()
                    
                    # Traders info
                    elif 'Traders:' in clean_line:
                        parts = clean_line.split('Traders:')[1].split('(')
                        if len(parts) > 0:
                            data['traders']['count'] = int(parts[0].strip())
                        if len(parts) > 1:
                            last_swap = parts[1].split(')')[0].replace('last swap:', '').strip()
                            data['traders']['last_swap'] = last_swap
                    
                    # Holders info
                    elif 'Holders:' in clean_line and 'T10' in clean_line:
                        # Total holders
                        holders_match = re.search(r'Holders: (\d+)', clean_line)
                        if holders_match:
                            data['holders']['total'] = int(holders_match.group(1))
                        
                        # Top percentages
                        if 'T10' in clean_line:
                            t10_match = re.search(r'T10 ([\d.]+)', clean_line)
                            if t10_match:
                                data['holders']['top10_percentage'] = float(t10_match.group(1))
                        
                        if 'T25' in clean_line:
                            t25_match = re.search(r'T25 ([\d.]+)', clean_line)
                            if t25_match:
                                data['holders']['top25_percentage'] = float(t25_match.group(1))
                        
                        if 'T50' in clean_line:
                            t50_match = re.search(r'T50 ([\d.]+)', clean_line)
                            if t50_match:
                                data['holders']['top50_percentage'] = float(t50_match.group(1))
                    
                    # Dev Holds
                    elif 'Dev Holds:' in clean_line:
                        holds_match = re.search(r'Dev Holds: (\d+)', clean_line)
                        if holds_match:
                            data['dev_holds'] = int(holds_match.group(1))
                    
                    # Dev Sold
                    elif 'Dev Sold:' in clean_line:
                        sold_match = re.search(r'Dev Sold: (\d+) time.*?(\d+\.?\d*) SOL.*?(\d+\.?\d*)%', clean_line)
                        if sold_match:
                            data['dev_sold']['times'] = int(sold_match.group(1))
                            data['dev_sold']['sol'] = float(sold_match.group(2))
                            data['dev_sold']['percentage'] = float(sold_match.group(3))

                    # Dev bought info
                    elif 'Dev bought' in clean_line:
                        tokens_match = re.search(r'Dev bought ([\d.]+)([KMB]) tokens', clean_line)
                        sol_match = re.search(r'([\d.]+) SOL', clean_line)
                        percentage_match = re.search(r'([\d.]+)%', clean_line)
                        curve_match = re.search(r'\(([\d.]+)% of curve\)', clean_line)
                        
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

                # Dev bought info
                    if 'Dev bought' in clean_line:
                        tokens_match = re.search(r'(\d+\.?\d*)([KMB]) tokens', clean_line)
                        sol_match = re.search(r'(\d+\.?\d*) SOL', clean_line)
                        percentage_match = re.search(r'(\d+\.?\d*)%', clean_line)
                        curve_match = re.search(r'(\d+\.?\d*)% of curve', clean_line)
                        
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
                    if '🚩' in clean_line and 'Bundled' in clean_line:
                        count_match = re.search(r'(\d+) trades', clean_line)
                        supply_match = re.search(r'(\d+\.?\d*)%', clean_line)
                        curve_match = re.search(r'\((\d+\.?\d*)% of curve\)', clean_line)
                        sol_match = re.search(r'(\d+\.?\d*) SOL', clean_line)
                        
                        if count_match:
                            data['bundle']['count'] = int(count_match.group(1))
                        if supply_match:
                            data['bundle']['supply_percentage'] = float(supply_match.group(1))
                        if curve_match:
                            data['bundle']['curve_percentage'] = float(curve_match.group(1))
                        if sol_match:
                            data['bundle']['sol'] = float(sol_match.group(1))
                    
                    # Notable bundle info (📦 notable bundle(s))
                    if '📦' in clean_line and 'notable bundle' in clean_line:
                        clean_text = re.sub(r'\(http[^)]+\),', '', clean_line)
                        
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
                    if '🎯' in clean_line and 'Notable sniper activity' in clean_line:
                        tokens_match = re.search(r'(\d+\.?\d*)M', clean_line)
                        percentage_match = re.search(r'\((\d+\.?\d*)%\)', clean_line)
                        sol_match = re.search(r'(\d+\.?\d*) SOL', clean_line)
                        
                        if tokens_match:
                            data['sniper_activity']['tokens'] = float(tokens_match.group(1)) * 1000000
                        if percentage_match:
                            data['sniper_activity']['percentage'] = float(percentage_match.group(1))
                        if sol_match:
                            data['sniper_activity']['sol'] = float(sol_match.group(1))
                    
                    # Dev created tokens
                    elif 'Dev created' in clean_line:
                        match = re.search(r'Dev created (\d+)', clean_line)
                        if match:
                            data['dev_created_tokens'] = int(match.group(1))
                    
                                        # Same name count
                    elif 'same as' in clean_line and 'name' in clean_line.lower():
                        match = re.search(r'same as (\d+)', clean_line)
                        if match:
                            data['same_name_count'] = int(match.group(1))
                    
                    # Same website count
                    elif 'same as' in clean_line and 'website' in clean_line.lower():
                        match = re.search(r'same as (\d+)', clean_line)
                        if match:
                            data['same_website_count'] = int(match.group(1))
                    
                    # Same telegram count
                    elif 'same as' in clean_line and 'telegram' in clean_line.lower():
                        match = re.search(r'same as (\d+)', clean_line)
                        if match:
                            data['same_telegram_count'] = int(match.group(1))
                    
                    # Same twitter count
                    elif 'same as' in clean_line and 'twitter' in clean_line.lower():
                        match = re.search(r'same as (\d+)', clean_line)
                        if match:
                            data['same_twitter_count'] = int(match.group(1))

                except Exception as e:
                    logger.warning(f"Error parsing line '{line}': {str(e)}")
                    continue

            return data

        except Exception as e:
            logger.error(f"Error parsing Syrax Scanner message: {e}")
            return {
                'error': f"Parsing error: {str(e)}",
                'dev_bought': {'tokens': 'N/A', 'sol': 'N/A', 'percentage': 'N/A', 'curve_percentage': 'N/A'},
                'dev_created_tokens': 'N/A',
                'same_name_count': 'N/A',
                'same_website_count': 'N/A',
                'same_telegram_count': 'N/A',
                'same_twitter_count': 'N/A',
                'bundle': {'count': 'N/A', 'supply_percentage': 'N/A', 'curve_percentage': 'N/A', 'sol': 'N/A'},
                'notable_bundle': {'count': 'N/A', 'supply_percentage': 'N/A', 'curve_percentage': 'N/A', 'sol': 'N/A'},
                'sniper_activity': {'tokens': 'N/A', 'percentage': 'N/A', 'sol': 'N/A'},
                # Nauji laukai
                'created_time': 'N/A',
                'traders': {'count': 'N/A', 'last_swap': 'N/A'},
                'holders': {
                    'total': 'N/A',
                    'top10_percentage': 'N/A',
                    'top25_percentage': 'N/A',
                    'top50_percentage': 'N/A'
                },
                'dev_holds': 'N/A',
                'dev_sold': {'times': 'N/A', 'sol': 'N/A', 'percentage': 'N/A'}
            }

    async def parse_proficy_price(self, message: str) -> Dict:
        """Parse ProficyPriceBot message"""
        try:
            # 1. PAKEISTI: Inicializuojame data su None reikšmėmis vietoj 'N/A'
            data = {
                '5m': {
                    'price_change': None,
                    'volume': None,
                    'bs_ratio': None
                },
                '1h': {
                    'price_change': None,
                    'volume': None,
                    'bs_ratio': None
                }
            }

            # 2. PRIDĖTI: Patikrinimas ar message nėra tuščias
            if not message:
                logger.warning("Empty message received from ProficyPriceBot")
                return data
                
            lines = message.split('\n')
            
            # Šios funkcijos nekeičiamos
            def convert_volume(volume_str: str) -> float:
                """Helper function to convert volume with K or M suffix"""
                try:
                    volume_str = volume_str.replace('$', '')
                    if 'M' in volume_str:
                        return float(volume_str.replace('M', '')) * 1000000
                    elif 'K' in volume_str:
                        return float(volume_str.replace('K', '')) * 1000
                    return float(volume_str)
                except (ValueError, TypeError):
                    return 0

            def format_price_change(price: float) -> str:
                """Helper function to format price change with K suffix if needed"""
                if abs(price) >= 1000:
                    return f"{price/1000:.1f}K"
                return f"{price:.2f}"
            
            for line in lines:
                if 'Price' in line and 'Volume' in line and 'B/S' in line:
                    continue
                    
                # 3. PAKEISTI: Pridėti parts ilgio patikrinimą
                if '5M:' in line:
                    parts = line.split()
                    if len(parts) >= 4:  # Patikrinam ar užtenka dalių
                        data['5m'] = {
                            'price_change': convert_volume(parts[1].replace('%', '')),
                            'volume': convert_volume(parts[2]),
                            'bs_ratio': parts[3]
                        }
                    else:
                        logger.warning("Incomplete 5M data in ProficyPriceBot message")
                    
                if '1H:' in line:
                    parts = line.split()
                    if len(parts) >= 4:  # Patikrinam ar užtenka dalių
                        data['1h'] = {
                            'price_change': convert_volume(parts[1].replace('%', '')),
                            'volume': convert_volume(parts[2]),
                            'bs_ratio': parts[3]
                        }
                    else:
                        logger.warning("Incomplete 1H data in ProficyPriceBot message")

            # 4. PRIDĖTI: Patikrinimas ar gavome bent kokius nors duomenis
            if data['5m']['price_change'] is None and data['1h']['price_change'] is None:
                logger.warning("No valid intervals found in ProficyPriceBot message")
                    
            return data
            
        except Exception as e:
            # 5. PAKEISTI: Grąžinti None reikšmes vietoj 'N/A'
            logger.error(f"Error parsing ProficyPrice message: {e}")
            return {
                '5m': {
                    'price_change': None,
                    'volume': None,
                    'bs_ratio': None
                },
                '1h': {
                    'price_change': None,
                    'volume': None,
                    'bs_ratio': None
                }
            }

    
class MLIntervalAnalyzer:
    """ML klasė pirminių intervalų nustatymui"""
    def __init__(self):
        self.primary_features = [
            # Syrax Scanner parametrai (syrax_scanner_data lentelė)
            'dev_created_tokens',
            'same_name_count',
            'same_website_count',
            'same_telegram_count',
            'same_twitter_count',
            'dev_bought_percentage',
            'dev_bought_curve_percentage',
            'dev_sold_percentage',
            'holders_total',
            'holders_top10_percentage',
            'holders_top25_percentage',
            'holders_top50_percentage',
            
            # Soul Scanner parametrai (soul_scanner_data lentelė)
            'market_cap',
            'liquidity_usd',
            'mint_status',      # čia Boolean tipo
            'freeze_status',    # čia Boolean tipo
            'lp_status',       # čia Boolean tipo
            'total_scans',
            
            # Proficy parametrai (proficy_price_data lentelė)
            'volume_1h',
            'price_change_1h',
            'bs_ratio_1h',
            
            # Kiti parametrai iš syrax_scanner_data
            'bundle_count',
            'sniper_activity_tokens',
            'traders_count',
            'sniper_activity_percentage',
            'notable_bundle_supply_percentage',
            'bundle_supply_percentage'
        ]
        
        self.filter_status = {
                    'dev_created_tokens': False,
                    'same_name_count': False, 
                    'same_website_count': False,
                    'same_telegram_count': False,
                    'same_twitter_count': False,
                    'dev_bought_percentage': False,
                    'dev_bought_curve_percentage': False,
                    'dev_sold_percentage': False,
                    'holders_total': True,
                    'holders_top10_percentage': False,
                    'holders_top25_percentage': False,
                    'holders_top50_percentage': False,
                    'market_cap': True,
                    'liquidity_usd': False,
                    'mint_status': False,
                    'freeze_status': False,
                    'lp_status': False,
                    'total_scans': False,
                    'volume_1h': False,
                    'price_change_1h': False,
                    'bs_ratio_1h': False,
                    'bundle_count': False,
                    'sniper_activity_tokens': False,
                    'traders_count': True,
                    'sniper_activity_percentage': True,
                    'notable_bundle_supply_percentage': True,
                    'bundle_supply_percentage': True
                }
        
        self.scaler = MinMaxScaler()
        self.isolation_forest = IsolationForest(contamination='auto', random_state=42)
        self.intervals = {feature: {'min': float('inf'), 'max': float('-inf')} for feature in self.primary_features}

        # IQR daugikliai skirtingiems parametrams
        self.IQR_MULTIPLIERS = {
            'price_change_1h': 1.5,      # Mažesnis daugiklis kainų pokyčiams
            'market_cap': 1.5,           # Vidutinis daugiklis market cap
            'volume_1h': 1.5,            # Vidutinis daugiklis volume
            'holders_total': 1.5,        # Vidutinis daugiklis holders
            'total_scans': 1.5,          # Vidutinis daugiklis skanavimams
            'traders_count': 1.5,        # Vidutinis daugiklis traders
            'bs_ratio_1h': 1.5,          # Mažesnis daugiklis buy/sell ratio
            'liquidity_usd': 1.5,        # Vidutinis daugiklis likvidumui
            # Naujai pridedami parametrai
            'sniper_activity_percentage': 1.5,      # Vidutinis daugiklis sniper activity
            'notable_bundle_supply_percentage': 1.5, # Vidutinis daugiklis notable bundle
            'bundle_supply_percentage': 1.5,        # Vidutinis daugiklis bundle
    
                   
            'default': 2.5               # Visiems kitiems
        }

        # Absoliučios ribos parametrams
        self.ABSOLUTE_LIMITS = {
            'price_change_1h': (-100, 1000),      # nuo -100% iki 1000%
            'market_cap': (100, 1000000),      # nuo 100$ iki 1B$
            'volume_1h': (10, 10000000),          # nuo 10$ iki 10M$
            'holders_total': (100, 100000),         # nuo 1 iki 100k
            'liquidity_usd': (0, 10000000),      # nuo 10$ iki 10M$
            'total_scans': (10, 1000000),          # negali būti 0
            'traders_count': (100, 100000),         # negali būti 0
            'bundle_count': (0, 0),               # visada 0
            'mint_status': (0, 0),                # visada false (0)
            'freeze_status': (0, 0),              # visada false (0)
            'lp_status': (1, 1),                  # visada true (1)
            'sniper_activity_tokens': (0, 0),     # visada 0
            'bs_ratio_1h': (0.1, 10.0),           # nuo 0.1 iki 10.0
            # Naujai pridedami parametrai
            'sniper_activity_percentage': (0, 16),       # nuo 0% iki 100%
            'notable_bundle_supply_percentage': (0, 16),  # nuo 0% iki 100%
            'bundle_supply_percentage': (0, 16)          # nuo 0% iki 100%
        }

    def _parse_ratio_value(self, ratio_str) -> float:
        """Konvertuoja bs_ratio į float reikšmę
        
        Args:
            ratio_str: Gali būti string "X/Y" formatu arba skaičius
                
        Returns:
            float: Pirkimų/pardavimų santykis (buys/sells)
        """
        # Jei paduotas skaičius
        if isinstance(ratio_str, (int, float)):
            return float(ratio_str) if ratio_str > 0 else 1.0
            
        try:
            buys, sells = ratio_str.split('/')
            
            # Konvertuojame K į tūkstančius
            def convert_k(val: str) -> float:
                val = val.strip()
                if 'K' in val:
                    return float(val.replace('K', '')) * 1000
                return float(val)
            
            buys = convert_k(buys)
            sells = convert_k(sells)
            
            # Grąžiname tikrąjį santykį buys/sells
            if sells == 0:
                return 1.0  # Jei nėra pardavimų, grąžiname 1
                
            return buys / sells
            
        except (ValueError, TypeError, ZeroDivisionError, AttributeError):
            return 1.0  # Default santykis 1:1

    def validate_interval(self, feature: str, interval: dict) -> dict:
        """Validuoja ir koreguoja intervalą pagal absoliučias ribas"""
        if feature in self.ABSOLUTE_LIMITS:
            min_limit, max_limit = self.ABSOLUTE_LIMITS[feature]
            
            # Pritaikome absoliučias ribas
            interval['min'] = max(min_limit, interval['min'])
            interval['max'] = min(max_limit, interval['max'])
            
            # Jei tai parametras su fiksuota reikšme (0 arba 1)
            if min_limit == max_limit:
                interval['mean'] = min_limit
                interval['std'] = 0
                
        return interval

    def toggle_filter(self, filter_name: str, status: bool = None) -> bool:
        """
        Įjungia/išjungia filtrą arba toggle jei status nenurodyta
        """
        if filter_name not in self.filter_status:
            logger.warning(f"Filtras '{filter_name}' nerastas")
            return False
            
        if status is None:
            self.filter_status[filter_name] = not self.filter_status[filter_name]
        else:
            self.filter_status[filter_name] = status
            
        logger.info(f"Filtras '{filter_name}' nustatytas į {self.filter_status[filter_name]}")
        return self.filter_status[filter_name]

    def get_filter_status(self) -> Dict[str, bool]:
        """Grąžina visų filtrų statusą"""
        return self.filter_status.copy()

    def get_enabled_filters(self) -> List[str]:
        """Grąžina įjungtų filtrų sąrašą"""
        return [k for k, v in self.filter_status.items() if v]

    def get_disabled_filters(self) -> List[str]:
        """Grąžina išjungtų filtrų sąrašą"""
        return [k for k, v in self.filter_status.items() if not v]

        
        
    def calculate_intervals(self, successful_gems: List[Dict]):
        """Nustato intervalus naudojant ML iš sėkmingų GEM duomenų"""
        if not successful_gems or len(successful_gems) < Config.MIN_GEMS_FOR_ANALYSIS:
            logger.warning(f"Nepakanka duomenų ML intervalų nustatymui. Reikia bent {Config.MIN_GEMS_FOR_ANALYSIS} GEM'ų. Dabartinis kiekis: {len(successful_gems)}")
            return False
                
        # Toliau vykdome tik jei yra pakankamai duomenų
        X = []
        for gem in successful_gems:
            features = []
            for feature in self.primary_features:
                try:
                    if feature == 'bs_ratio_1h':
                        ratio_str = gem[feature] if gem[feature] is not None else '1/1'
                        value = self._parse_ratio_value(ratio_str)
                    else:
                        value = float(gem[feature] if gem[feature] is not None else 0)
                except (ValueError, TypeError):
                    value = 1.0 if feature == 'bs_ratio_1h' else 0.0
                features.append(value)
            X.append(features)
        
        X = np.array(X)
        
        # Apmokome Isolation Forest
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        
        # Nustatome intervalus kiekvienam parametrui
        for i, feature in enumerate(self.primary_features):
            values = X[:, i]
            predictions = self.isolation_forest.predict(X_scaled)
            normal_values = values[predictions == 1]
            
            if len(normal_values) > 0:
                # Statistinė analizė - ML nustato intervalus
                q1 = np.percentile(normal_values, 25)
                q3 = np.percentile(normal_values, 75)
                iqr = q3 - q1
                mean_val = np.mean(normal_values)
                std_val = np.std(normal_values)
                
                # Pasirenkame tinkamą daugiklį
                multiplier = self.IQR_MULTIPLIERS.get(feature, self.IQR_MULTIPLIERS['default'])
                
                # ML nustatytas intervalas
                interval = {
                    'min': q1 - multiplier * iqr,
                    'max': q3 + multiplier * iqr,
                    'mean': mean_val,
                    'std': std_val
                }
                
                # Tik po ML nustatymo, patikriname ar reikia taikyti ABSOLUTE_LIMITS
                if feature in self.ABSOLUTE_LIMITS:
                    min_limit, max_limit = self.ABSOLUTE_LIMITS[feature]
                    if min_limit == max_limit:
                        # Fixed value parametrams vis tiek naudojame absoliučias reikšmes
                        interval = {
                            'min': min_limit,
                            'max': max_limit,
                            'mean': min_limit,
                            'std': 0
                        }
                    else:
                        # Kitiems parametrams ABSOLUTE_LIMITS tik kaip saugiklis
                        if interval['min'] < min_limit:
                            interval['min'] = min_limit
                        if interval['max'] > max_limit:
                            interval['max'] = max_limit
                
                self.intervals[feature] = interval
        
        logger.info(f"ML intervalai atnaujinti sėkmingai su {len(successful_gems)} GEM'ais")
        return True
        
    def check_primary_parameters(self, token_data: Dict) -> Dict:
        """Tikrina ar token'o parametrai patenka į ML nustatytus intervalus"""
        results = {}

        # Tik vienas ciklas
        for feature in self.primary_features:
            # Praleidžiame išjungtus filtrus ir features kurių nėra filter_status
            if not self.filter_status.get(feature, False):
                # Įrašome default reikšmes į results, bet netikriname jų
                results[feature] = {
                    'value': 0,
                    'in_range': True,  # Defaultinės reikšmės visada "in range"
                    'z_score': 0,
                    'interval': self.intervals[feature]
                }
                continue
        
            try:
                # Tiesiogiai imame reikšmę iš parametrų
                if feature == 'bs_ratio_1h':
                    value = self._parse_ratio_value(token_data.get(feature, '1/1'))
                elif feature in ['mint_status', 'freeze_status', 'lp_status']:
                    # Boolean laukams naudojame False kai None
                    value = bool(token_data.get(feature, False))
                else:
                    # Visiems kitiems skaitiniams laukams
                    value = float(token_data.get(feature, 0))
                    
                    # Svarbi dev_sold_percentage patikra
                    if feature == 'dev_sold_percentage':
                        dev_bought = float(token_data.get('dev_bought_percentage', 0))
                        # Jei dev'as pirko bet dar nepardavė - rizikinga situacija
                        if dev_bought > 0 and value == 0:
                            results[feature] = {
                                'value': value,
                                'in_range': False,
                                'z_score': float('inf'),
                                'interval': self.intervals[feature]
                            }
                            continue
                                
            except (ValueError, TypeError, KeyError):
                value = 1.0 if feature == 'bs_ratio_1h' else 0.0
                
            interval = self.intervals[feature]
            
            in_range = interval['min'] <= value <= interval['max']
            z_score = abs((value - interval['mean']) / interval['std']) if interval['std'] > 0 else float('inf')
            
            results[feature] = {
                'value': value,
                'in_range': in_range,
                'z_score': z_score,
                'interval': interval
            }
            
        # Bendras rezultatas - tikriname TIK įjungtų filtrų rezultatus (PO ciklo)
        active_results = {k: v for k, v in results.items() if self.filter_status.get(k, True)}
        all_in_range = all(result['in_range'] for result in active_results.values())
        
        # Patikriname ar yra ne-inf z_scores
        finite_z_scores = [result['z_score'] for result in active_results.values() if result['z_score'] != float('inf')]
        
        # Jei yra baigtinių z_scores, skaičiuojame jų vidurkį, jei ne - grąžiname inf
        avg_z_score = np.mean(finite_z_scores) if finite_z_scores else float('inf')

        return {
            'passed': all_in_range,
            'avg_z_score': avg_z_score,
            'details': results
        }

class MLGEMAnalyzer:
    def __init__(self):
        """Inicializuoja ML GEM analizatorių"""
        self.interval_analyzer = MLIntervalAnalyzer()
        self.scaler = MinMaxScaler()
        self.isolation_forest = IsolationForest(contamination='auto', random_state=42)
        self.db = DatabaseManager()
        
        # Apibrėžiame pagrindinius parametrus analizei
        self.primary_features = [
            'dev_created_tokens', 'same_name_count', 'same_website_count',
            'same_telegram_count', 'same_twitter_count', 'dev_bought_percentage',
            'dev_bought_curve_percentage', 'dev_sold_percentage', 'holders_total',
            'holders_top10_percentage', 'holders_top25_percentage',
            'holders_top50_percentage', 'market_cap', 'liquidity_usd',
            'volume_1h', 'price_change_1h', 'bs_ratio_1h'
        ]
        
        # Apibrėžiame visus ML features pagal scannerius
        self.features = {
            'soul': [
                'market_cap', 'ath_market_cap', 'liquidity_usd', 'liquidity_sol',
                'mint_status', 'freeze_status', 'lp_status', 'dex_status_paid', 
                'dex_status_ads', 'total_scans'
            ],
            'syrax': [
                'dev_bought_tokens', 'dev_bought_sol', 'dev_bought_percentage',
                'dev_bought_curve_percentage', 'dev_created_tokens',
                'same_name_count', 'same_website_count', 'same_telegram_count',
                'same_twitter_count', 'bundle_count', 'bundle_supply_percentage',
                'bundle_curve_percentage', 'bundle_sol', 'notable_bundle_count',
                'notable_bundle_supply_percentage', 'notable_bundle_curve_percentage',
                'notable_bundle_sol', 'sniper_activity_tokens',
                'sniper_activity_percentage', 'sniper_activity_sol',
                'holders_total', 'holders_top10_percentage',
                'holders_top25_percentage', 'holders_top50_percentage',
                'dev_holds', 'dev_sold_times', 'dev_sold_sol', 'dev_sold_percentage'
            ],
            'proficy': [
                'price_change_5m', 'volume_5m', 'bs_ratio_5m',
                'price_change_1h', 'volume_1h', 'bs_ratio_1h'
            ]
        }
        
        self.gem_tokens = []
        self.load_gem_data()

    def load_gem_data(self):
        """Užkrauna GEM duomenis iš DB ir apmoko modelius"""
        try:
            print("\n=== Loading GEM Data ===")
            # Gauname duomenis iš DB
            self.gem_tokens = self.db.load_gem_tokens()
            print(f"Loaded {len(self.gem_tokens)} GEM tokens from database")
            
            if self.gem_tokens:
                print("\nFirst GEM token data example:")
                first_token = self.gem_tokens[0]
                print(f"Address: {first_token.get('address')}")
                print(f"Name: {first_token.get('name')}")
                print(f"Market Cap: {first_token.get('market_cap')}")
                
                # Apmokome modelius
                self.interval_analyzer.calculate_intervals(self.gem_tokens)
                success = self._train_main_model()
                print(f"Models trained successfully: {success}")
            else:
                print("WARNING: No GEM tokens found in database!")
                
        except Exception as e:
            print(f"ERROR loading GEM data: {str(e)}")
            logger.error(f"Error loading GEM data: {e}")

    def _train_main_model(self):
        """Apmoko pagrindinį ML modelį su visais parametrais"""
        try:
            if not self.gem_tokens or len(self.gem_tokens) < 3:
                print("Not enough GEM tokens for training (minimum 3 required)")
                return False

            print("\n=== Training Main Model ===")
            X = self._prepare_training_data()
            if len(X) > 0:
                print(f"Training with {len(X)} samples")
                X_scaled = self.scaler.fit_transform(X)
                self.isolation_forest.fit(X_scaled)
                print("Model training completed successfully")
                return True
            return False
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False

    def _safe_float(self, value):
        """Saugiai konvertuoja reikšmę į float"""
        if value is None or value == 'N/A' or value == '':
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _prepare_training_data(self):
        """Paruošia duomenis ML modelio apmokymui"""
        print("\n=== Preparing Training Data ===")
        data = []
        try:
            for token in self.gem_tokens:
                features = []
                # Soul features
                for feature in self.features['soul']:
                    value = self._safe_float(token.get(feature))
                    features.append(value)

                # Syrax features    
                for feature in self.features['syrax']:
                    value = self._safe_float(token.get(feature))
                    features.append(value)

                # Proficy features    
                for feature in self.features['proficy']:
                    if 'bs_ratio' in feature:
                        value = self._parse_bs_ratio(token.get(feature, '1/1'))
                    else:
                        value = self._safe_float(token.get(feature))
                    features.append(value)
                
                data.append(features)
            
            print(f"Successfully prepared {len(data)} training samples")
            return np.array(data)
        except Exception as e:
            print(f"Error preparing training data: {str(e)}")
            return np.array([])

    def analyze_token(self, token_data: Dict) -> Dict:
        """
        Pilna token'o analizė naudojant duomenis iš DB
        
        Args:
            token_data: Token duomenų dictionary su soul, syrax ir proficy sekcijomis
            
        Returns:
            Dict: Analizės rezultatai
        """
        print("\n=== Starting Token Analysis ===")
        print(f"Available GEM tokens for analysis: {len(self.gem_tokens)}")
        
        try:
            # Pirma patikriname ar turime pakankamai GEM duomenų ir ar modelis apmokytas
            if len(self.gem_tokens) < Config.MIN_GEMS_FOR_ANALYSIS:
                return {
                    'status': 'pending',
                    'message': f'Reikia daugiau GEM duomenų (min: {Config.MIN_GEMS_FOR_ANALYSIS}, current: {len(self.gem_tokens)})',
                    'collected_gems': len(self.gem_tokens)
                }

            # SVARBU: Apmokome modelį prieš analizę
            if not self._train_main_model():
                return {
                    'status': 'failed',
                    'stage': 'training',
                    'message': 'Failed to train ML model'
                }

            # Gauname adresą iš soul sekcijos
            if 'soul' in token_data:
                address = token_data['soul'].get('contract_address')
            else:
                print(f"Error: Missing 'soul' section in token_data")
                return {
                    'status': 'failed',
                    'stage': 'validation',
                    'message': 'Missing soul scanner data'
                }

            if not address:
                print(f"Error: No contract_address in soul section")
                return {
                    'status': 'failed',
                    'stage': 'validation',
                    'message': 'No contract address found'
                }

            print(f"Analyzing token: {address}")
        
            # Tikriname ar turime pakankamai GEM duomenų
            if len(self.gem_tokens) < Config.MIN_GEMS_FOR_ANALYSIS:
                return {
                    'status': 'pending',
                    'message': f'Reikia daugiau GEM duomenų (min: {Config.MIN_GEMS_FOR_ANALYSIS}, current: {len(self.gem_tokens)})',
                    'collected_gems': len(self.gem_tokens)
                }
            # Gauname visus token duomenis iš DB
            self.db.cursor.execute('''
                SELECT 
                    -- Soul Scanner duomenys
                    s.name,
                    s.symbol,
                    s.market_cap,
                    s.ath_market_cap,
                    s.liquidity_usd,
                    s.liquidity_sol,
                    CAST(s.mint_status AS INTEGER) as mint_status,
                    CAST(s.freeze_status AS INTEGER) as freeze_status,
                    CAST(s.lp_status AS INTEGER) as lp_status,
                    CAST(s.dex_status_paid AS INTEGER) as dex_status_paid,
                    CAST(s.dex_status_ads AS INTEGER) as dex_status_ads,
                    s.total_scans,
                    s.social_link_x,
                    s.social_link_tg,
                    s.social_link_web,
                    
                    -- Syrax Scanner duomenys
                    sy.dev_bought_tokens,
                    sy.dev_bought_sol,
                    sy.dev_bought_percentage,
                    sy.dev_bought_curve_percentage,
                    sy.dev_created_tokens,
                    sy.same_name_count,
                    sy.same_website_count,
                    sy.same_telegram_count,
                    sy.same_twitter_count,
                    sy.bundle_count,
                    sy.bundle_supply_percentage,
                    sy.bundle_curve_percentage,
                    sy.bundle_sol,
                    sy.notable_bundle_count,
                    sy.notable_bundle_supply_percentage,
                    sy.notable_bundle_curve_percentage,
                    sy.notable_bundle_sol,
                    sy.sniper_activity_tokens,
                    sy.sniper_activity_percentage,
                    sy.sniper_activity_sol,
                    sy.created_time,
                    sy.traders_count,
                    sy.traders_last_swap,
                    sy.holders_total,
                    sy.holders_top10_percentage,
                    sy.holders_top25_percentage,
                    sy.holders_top50_percentage,
                    sy.dev_holds,
                    sy.dev_sold_times,
                    sy.dev_sold_sol,
                    sy.dev_sold_percentage,
                    
                    -- Proficy Price duomenys
                    p.price_change_5m,
                    p.volume_5m,
                    p.bs_ratio_5m,
                    p.price_change_1h,
                    p.volume_1h,
                    p.bs_ratio_1h
                FROM tokens t
                JOIN soul_scanner_data s ON t.address = s.token_address
                JOIN syrax_scanner_data sy ON t.address = sy.token_address
                JOIN proficy_price_data p ON t.address = p.token_address
                WHERE t.address = ?
                ORDER BY s.scan_time DESC, sy.scan_time DESC, p.scan_time DESC
                LIMIT 1
            ''', (address,))

            row = self.db.cursor.fetchone()
            if not row:
                print(f"Error: No data found for token {address}")
                return {
                    'status': 'failed',
                    'stage': 'data',
                    'message': f'Token data not found in database for address: {address}'
                }

            # Konvertuojame į dictionary
            db_data = dict(row)
            
            # Debug - spausdiname gautus duomenis
            print("\nToken Data from Database:")
            print(json.dumps(db_data, indent=2))

            try:
                # Suformuojame primary check duomenis
                primary_data = {
                    'dev_created_tokens': float(db_data.get('dev_created_tokens', 0)),
                    'same_name_count': float(db_data.get('same_name_count', 0)),
                    'same_website_count': float(db_data.get('same_website_count', 0)),
                    'same_telegram_count': float(db_data.get('same_telegram_count', 0)),
                    'same_twitter_count': float(db_data.get('same_twitter_count', 0)),
                    'dev_bought_percentage': float(db_data.get('dev_bought_percentage', 0)),
                    'dev_bought_curve_percentage': float(db_data.get('dev_bought_curve_percentage', 0)),
                    'dev_sold_percentage': float(db_data.get('dev_sold_percentage', 0)),
                    'holders_total': float(db_data.get('holders_total', 0)),
                    'holders_top10_percentage': float(db_data.get('holders_top10_percentage', 0)),
                    'holders_top25_percentage': float(db_data.get('holders_top25_percentage', 0)),
                    'holders_top50_percentage': float(db_data.get('holders_top50_percentage', 0)),
                    'market_cap': float(db_data.get('market_cap', 0)),
                    'liquidity_usd': float(db_data.get('liquidity_usd', 0)),
                    'volume_1h': float(db_data.get('volume_1h', 0)),
                    'price_change_1h': float(db_data.get('price_change_1h', 0)),
                    'bs_ratio_1h': self._parse_bs_ratio(db_data.get('bs_ratio_1h', '1/1')),
                    'mint_status': float(db_data.get('mint_status', 0)),
                    'freeze_status': float(db_data.get('freeze_status', 0)),
                    'lp_status': float(db_data.get('lp_status', 0)),
                    'total_scans': float(db_data.get('total_scans', 0)),
                    'bundle_count': float(db_data.get('bundle_count', 0)),
                    'sniper_activity_tokens': float(db_data.get('sniper_activity_tokens', 0)),
                    'traders_count': float(db_data.get('traders_count', 0))
                }

                # Pirminė parametrų patikra
                primary_check = self.interval_analyzer.check_primary_parameters(primary_data)
                print("\nPrimary Check Results:")
                for param, details in primary_check['details'].items():
                    print(f"{param}:")
                    print(f"  Value: {details['value']}")
                    print(f"  In Range: {details['in_range']}")
                    print(f"  Z-Score: {details['z_score']}")

                if not primary_check['passed']:
                    return {
                        'status': 'failed',
                        'stage': 'primary',
                        'score': 0,
                        'details': primary_check['details'],
                        'message': 'Token nepraėjo pirminės filtracijos'
                    }

                # ML analizei ruošiame features pagal scannerius
                try:
                    features = []
                    feature_details = {}
                    
                    # Soul scanner features
                    soul_features = {
                        'market_cap': float(db_data.get('market_cap', 0)),
                        'ath_market_cap': float(db_data.get('ath_market_cap', 0)),
                        'liquidity_usd': float(db_data.get('liquidity_usd', 0)),
                        'liquidity_sol': float(db_data.get('liquidity_sol', 0)),
                        'mint_status': float(db_data.get('mint_status', 0)),
                        'freeze_status': float(db_data.get('freeze_status', 0)),
                        'lp_status': float(db_data.get('lp_status', 0)),
                        'dex_status_paid': float(db_data.get('dex_status_paid', 0)),
                        'dex_status_ads': float(db_data.get('dex_status_ads', 0)),
                        'total_scans': float(db_data.get('total_scans', 0))
                    }
                    features.extend(soul_features.values())
                    feature_details['soul'] = soul_features

                    # Syrax scanner features
                    syrax_features = {
                        'dev_bought_tokens': float(db_data.get('dev_bought_tokens', 0)),
                        'dev_bought_sol': float(db_data.get('dev_bought_sol', 0)),
                        'dev_bought_percentage': float(db_data.get('dev_bought_percentage', 0)),
                        'dev_bought_curve_percentage': float(db_data.get('dev_bought_curve_percentage', 0)),
                        'dev_created_tokens': float(db_data.get('dev_created_tokens', 0)),
                        'same_name_count': float(db_data.get('same_name_count', 0)),
                        'same_website_count': float(db_data.get('same_website_count', 0)),
                        'same_telegram_count': float(db_data.get('same_telegram_count', 0)),
                        'same_twitter_count': float(db_data.get('same_twitter_count', 0)),
                        'bundle_count': float(db_data.get('bundle_count', 0)),
                        'bundle_supply_percentage': float(db_data.get('bundle_supply_percentage', 0)),
                        'bundle_curve_percentage': float(db_data.get('bundle_curve_percentage', 0)),
                        'bundle_sol': float(db_data.get('bundle_sol', 0)),
                        'notable_bundle_count': float(db_data.get('notable_bundle_count', 0)),
                        'notable_bundle_supply_percentage': float(db_data.get('notable_bundle_supply_percentage', 0)),
                        'notable_bundle_curve_percentage': float(db_data.get('notable_bundle_curve_percentage', 0)),
                        'notable_bundle_sol': float(db_data.get('notable_bundle_sol', 0)),
                        'sniper_activity_tokens': float(db_data.get('sniper_activity_tokens', 0)),
                        'sniper_activity_percentage': float(db_data.get('sniper_activity_percentage', 0)),
                        'sniper_activity_sol': float(db_data.get('sniper_activity_sol', 0)),
                        'holders_total': float(db_data.get('holders_total', 0)),
                        'holders_top10_percentage': float(db_data.get('holders_top10_percentage', 0)),
                        'holders_top25_percentage': float(db_data.get('holders_top25_percentage', 0)),
                        'holders_top50_percentage': float(db_data.get('holders_top50_percentage', 0)),
                        'dev_holds': float(db_data.get('dev_holds', 0)),
                        'dev_sold_times': float(db_data.get('dev_sold_times', 0)),
                        'dev_sold_sol': float(db_data.get('dev_sold_sol', 0)),
                        'dev_sold_percentage': float(db_data.get('dev_sold_percentage', 0))
                    }
                    features.extend(syrax_features.values())
                    feature_details['syrax'] = syrax_features

                    # Proficy features
                    proficy_features = {
                        'price_change_5m': float(db_data.get('price_change_5m', 0)),
                        'volume_5m': float(db_data.get('volume_5m', 0)),
                        'bs_ratio_5m': self._parse_bs_ratio(db_data.get('bs_ratio_5m', '1/1')),
                        'price_change_1h': float(db_data.get('price_change_1h', 0)),
                        'volume_1h': float(db_data.get('volume_1h', 0)),
                        'bs_ratio_1h': self._parse_bs_ratio(db_data.get('bs_ratio_1h', '1/1'))
                    }
                    features.extend(proficy_features.values())
                    feature_details['proficy'] = proficy_features

                    # Debug - ištraukti features
                    print("\nExtracted Features:")
                    for scanner, features_dict in feature_details.items():
                        print(f"\n{scanner.upper()} Features:")
                        for feature, value in features_dict.items():
                            print(f"  {feature}: {value}")

                    # ML analizė
                    try:
                        X = np.array([features])
                        X_scaled = self.scaler.transform(X)
                        anomaly_score = self.isolation_forest.score_samples(X_scaled)[0]
                        similarity_score = (anomaly_score + 1) / 2 * 100

                        # Formuojame rezultatą
                        result = {
                            'status': 'success',
                            'stage': 'full',
                            'primary_check': primary_check,
                            'similarity_score': similarity_score,
                            'avg_z_score': primary_check['avg_z_score'],
                            'feature_analysis': feature_details,
                            'recommendation': self._generate_recommendation(similarity_score, primary_check['avg_z_score']),
                            'confidence_level': self._calculate_confidence(similarity_score, primary_check['avg_z_score'])
                        }

                        print("\nAnalysis Results:")
                        print(f"Similarity Score: {similarity_score:.2f}%")
                        print(f"Confidence Level: {result['confidence_level']:.2f}%")
                        print(f"Recommendation: {result['recommendation']}")

                        return result

                    except Exception as e:
                        print(f"Error during ML analysis: {str(e)}")
                        return {
                            'status': 'failed',
                            'stage': 'ml_analysis',
                            'message': str(e)
                        }

                except Exception as e:
                    print(f"Error during feature extraction: {str(e)}")
                    return {
                        'status': 'failed',
                        'stage': 'feature_extraction',
                        'message': str(e)
                    }

            except Exception as e:
                print(f"Error during primary check: {str(e)}")
                return {
                    'status': 'failed',
                    'stage': 'primary_check',
                    'message': str(e)
                }

        except Exception as e:
            print(f"Error during token analysis: {str(e)}")
            return {
                'status': 'failed',
                'stage': 'analysis',
                'message': f'Analysis error: {str(e)}'
            }
            
        finally:
            try:
                print(f"\n=== Analysis Complete ===")
                print(f"Timestamp (UTC): 2025-02-11 23:25:04")
                print(f"User: minijus05")
                print("="*50)
            except Exception as e:
                print(f"Error in cleanup: {str(e)}")
    
    def _parse_bs_ratio(self, ratio_str: str) -> float:
        """Konvertuoja B/S ratio string į float"""
        try:
            if not ratio_str or ratio_str == 'N/A':
                return 1.0
                
            if isinstance(ratio_str, str) and '/' in ratio_str:
                buy_str, sell_str = ratio_str.split('/')
                
                # Konvertuojame K į tūkstančius
                buy = float(buy_str.replace('K', '')) * 1000 if 'K' in buy_str else float(buy_str)
                sell = float(sell_str.replace('K', '')) * 1000 if 'K' in sell_str else float(sell_str)
                
                return buy / sell if sell != 0 else 1.0
                
            return float(ratio_str) if ratio_str else 1.0
        except:
            return 1.0

    def _generate_recommendation(self, similarity_score: float, z_score: float) -> str:
        """
        Generuoja rekomendaciją pagal panašumo rodiklį ir z-score
        
        Args:
            similarity_score: Panašumo į GEM score (0-100)
            z_score: Vidutinis Z-score iš pirminės patikros
            
        Returns:
            str: Rekomendacija
        """
        try:
            print("\n=== Generating Recommendation ===")
            print(f"Similarity Score: {similarity_score:.2f}")
            print(f"Average Z-Score: {z_score:.2f}")
            
            if similarity_score >= 80 and z_score < 1.5:
                return "STRONG GEM POTENTIAL"
            elif similarity_score >= 60 and z_score < 2:
                return "MODERATE GEM POTENTIAL"
            elif similarity_score >= 40:
                return "WEAK GEM POTENTIAL"
            return "NOT RECOMMENDED"
            
        except Exception as e:
            print(f"Error generating recommendation: {str(e)}")
            return "ERROR IN RECOMMENDATION"

    def _calculate_confidence(self, similarity_score: float, z_score: float) -> float:
        """
        Apskaičiuoja pasitikėjimo lygį rekomendacija
        
        Args:
            similarity_score: Panašumo į GEM score (0-100)
            z_score: Vidutinis Z-score iš pirminės patikros
            
        Returns:
            float: Pasitikėjimo lygis (0-100)
        """
        try:
            print("\n=== Calculating Confidence Level ===")
            
            # Normalizuojame similarity_score į 0-1
            norm_similarity = similarity_score / 100
            
            # Apskaičiuojame z-score įtaką (inverse relationship)
            z_score_impact = 1 / (1 + abs(z_score))
            
            # Skaičiuojame bendrą pasitikėjimo lygį
            confidence = norm_similarity * z_score_impact * 100
            
            # Apribojame rezultatą tarp 0 ir 100
            confidence = min(max(confidence, 0), 100)
            
            print(f"Calculated Confidence: {confidence:.2f}%")
            return confidence
            
        except Exception as e:
            print(f"Error calculating confidence: {str(e)}")
            return 0.0

    def add_gem_token(self, token_data: Dict):
        """
        Prideda naują GEM token'ą į ML modelio duomenis ir atnaujina modelius
        
        Args:
            token_data: Token'o duomenys iš duomenų bazės
        """
        try:
            print("\n=== Adding GEM Token to ML Model ===")
            print(f"Token Address: {token_data.get('address')}")
            
            # Gauname pilnus token duomenis iš DB
            self.db.cursor.execute('''
                SELECT 
                    t.address,
                    -- Soul Scanner duomenys
                    s.name,
                    s.symbol,
                    s.market_cap,
                    s.ath_market_cap,
                    s.liquidity_usd,
                    s.liquidity_sol,
                    CAST(s.mint_status AS INTEGER) as mint_status,
                    CAST(s.freeze_status AS INTEGER) as freeze_status,
                    CAST(s.lp_status AS INTEGER) as lp_status,
                    CAST(s.dex_status_paid AS INTEGER) as dex_status_paid,
                    CAST(s.dex_status_ads AS INTEGER) as dex_status_ads,
                    s.total_scans,
                    -- Syrax Scanner duomenys
                    sy.*,
                    -- Proficy duomenys
                    p.*
                FROM tokens t
                JOIN soul_scanner_data s ON t.address = s.token_address
                JOIN syrax_scanner_data sy ON t.address = sy.token_address
                JOIN proficy_price_data p ON t.address = p.token_address
                WHERE t.address = ?
                ORDER BY s.scan_time ASC, sy.scan_time ASC, p.scan_time ASC
                LIMIT 1
            ''', (token_data.get('address'),))
            
            db_data = dict(self.db.cursor.fetchone())
            
            # Pridedame į gem_tokens sąrašą ML analizei
            if db_data and db_data not in self.gem_tokens:
                self.gem_tokens.append(db_data)
                
                # Perskaičiuojame intervalus
                self.interval_analyzer.calculate_intervals(self.gem_tokens)
                
                # Permokiname modelį
                self._train_main_model()
                
                print("GEM token added to ML model and models updated successfully")
            else:
                print("Token already exists in ML model or data not found")
                
        except Exception as e:
            print(f"Error adding GEM token to ML model: {str(e)}")
            logger.error(f"Failed to add GEM token to ML model: {e}")

    def __str__(self):
        """String reprezentacija debuginimui"""
        return f"MLGEMAnalyzer(gems={len(self.gem_tokens)}, features={sum(len(f) for f in self.features.values())})"

    def __repr__(self):
        """Reprezentacija debuginimui"""
        return self.__str__()

    
            
class CustomSQLiteSession(SQLiteSession):
    def __init__(self, session_id):
        super().__init__(session_id)
        self._db_connection = None
        self._db_cursor = None
        self._connect()

    def _connect(self):
        if self._db_connection is None:
            self._db_connection = sqlite3.connect(self.filename, timeout=30.0)
            self._db_cursor = self._db_connection.cursor()

    def close(self):
        if self._db_cursor:
            self._db_cursor.close()
        if self._db_connection:
            self._db_connection.close()
        self._db_cursor = None
        self._db_connection = None

    def get_cursor(self):
        """Returns the current cursor or creates a new one"""
        if self._db_cursor is None:
            self._connect()
        return self._db_cursor

    def execute(self, *args, **kwargs):
        for attempt in range(5):
            try:
                cursor = self.get_cursor()
                cursor.execute(*args, **kwargs)
                self._db_connection.commit()
                return cursor
            except sqlite3.OperationalError as e:
                if 'database is locked' in str(e) and attempt < 4:
                    self.close()
                    time.sleep(1)
                    continue
                raise

    def fetchone(self):
        return self.get_cursor().fetchone()

    def fetchall(self):
        return self.get_cursor().fetchall()

    def commit(self):
        if self._db_connection:
            self._db_connection.commit()

    
async def main():
    """Main function to run the token monitor"""
    try:
        # Initialize custom sessions
        scanner_session = CustomSQLiteSession('scanner_session')
        monitor_session = CustomSQLiteSession('token_monitor_session')
        
        # Initialize token monitor with custom sessions
        monitor = TokenMonitor(monitor_session, scanner_session)
        
        for attempt in range(3):  # 3 bandymai inicializuoti
            try:
                await monitor.initialize()
                logger.info("Token monitor initialized successfully")
                break
            except sqlite3.OperationalError as e:
                if 'database is locked' in str(e) and attempt < 2:
                    logger.warning(f"Database locked, attempt {attempt + 1}/3. Waiting...")
                    time.sleep(2)
                    continue
                raise
            except Exception as e:
                logger.error(f"Initialization error: {e}")
                raise

        print(f"\nCurrent Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current User's Login: minijus05\n")

        # Rodyti duomenų bazės statistiką po inicializacijos
        #monitor.db.display_database_stats()

        @monitor.telegram.on(events.NewMessage(chats=Config.TELEGRAM_SOURCE_CHATS))
        async def message_handler(event):
            await monitor.handle_new_message(event)

        # Pridedame delete komandą
        @monitor.telegram.on(events.NewMessage(pattern='/delete'))
        async def delete_handler(event):
            await monitor.handle_delete_command(event)

        print("Bot started! Press Ctrl+C to stop.")
        
        await monitor.telegram.run_until_disconnected()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        # Cleanup
        try:
            await monitor.telegram.disconnect()
            await monitor.scanner_client.disconnect()
        except:
            pass
        raise
    finally:
        # Final cleanup
        try:
            scanner_session.close()
            monitor_session.close()
        except:
            pass
        
class DatabaseManager:
    def __init__(self, db_path='token_monitor.db'):
        self.db_path = db_path
        self._ensure_connection()
        

    def _ensure_connection(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Leidžia gauti rezultatus kaip žodynus
        self.cursor = self.conn.cursor()

    def calculate_multiplier(self, address: str, current_mc: float) -> tuple[float, float]:
        """
        Apskaičiuoja token'o multiplier'į lyginant su pradiniu Market Cap
        
        Args:
            address: Token'o adresas
            current_mc: Dabartinis Market Cap
        
        Returns:
            tuple[float, float]: (pradinis_mc, multiplier)
        """
        # Gauname pradinį Market Cap
        self.cursor.execute('''
            SELECT market_cap 
            FROM soul_scanner_data 
            WHERE token_address = ? 
            ORDER BY scan_time ASC 
            LIMIT 1
        ''', (address,))
        
        result = self.cursor.fetchone()
        if not result or not result[0] or result[0] == 0:
            return 0, 0
            
        initial_mc = result[0]
        multiplier = current_mc / initial_mc if current_mc > 0 else 0
        
        return initial_mc, multiplier

    def save_token_data(self, address: str, soul_data: Dict, syrax_data: Dict, proficy_data: Dict, is_new_token: bool):
        try:
            
            
            current_mc = soul_data.get('market_cap', 0) if soul_data else 0
            
            # LOGGER 2: Patikriname ar token'as jau egzistuoja
            self.cursor.execute("SELECT address FROM tokens WHERE address = ?", (address,))
            exists = self.cursor.fetchone() is not None
            
            
            # Pradedame transaction
            self.cursor.execute('BEGIN TRANSACTION')
            
            # LOGGER 3: Įterpiame naują token'ą jei reikia
            if is_new_token:
                try:
                    
                    self.cursor.execute('''
                        INSERT INTO tokens (address, first_seen, last_updated, is_gem, total_scans)
                        VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, FALSE, 1)
                    ''', (address,))
                    
                    
                    # Soul Scanner duomenų įrašymas
                    if soul_data:
                        
                        self.cursor.execute('''
                            INSERT INTO soul_scanner_data (
                                token_address, scan_time,
                                name, symbol, market_cap, ath_market_cap,
                                liquidity_usd, liquidity_sol, mint_status, freeze_status,
                                lp_status, dex_status_paid, dex_status_ads, total_scans,
                                social_link_x, social_link_tg, social_link_web
                            ) VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            address,
                            soul_data.get('name'),
                            soul_data.get('symbol'),
                            soul_data.get('market_cap'),
                            soul_data.get('ath_market_cap'),
                            soul_data.get('liquidity', {}).get('usd'),
                            soul_data.get('liquidity', {}).get('sol'),
                            soul_data.get('mint_status'),
                            soul_data.get('freeze_status'),
                            soul_data.get('lp_status'),
                            soul_data.get('dex_status', {}).get('paid'),
                            soul_data.get('dex_status', {}).get('ads'),
                            soul_data.get('total_scans'),
                            soul_data.get('social_links', {}).get('X'),
                            soul_data.get('social_links', {}).get('TG'),
                            soul_data.get('social_links', {}).get('WEB')
                        ))
                        
                    
                    # Syrax Scanner duomenų įrašymas
                    if syrax_data:
                        
                        #for key, value in syrax_data.items():
                            #print(f"- {key}: {value}")

                        try:
                            print(f"[DEBUG] Attempting to insert Syrax Scanner data for {address}")
                            self.cursor.execute('''
                                INSERT INTO syrax_scanner_data (
                                    token_address, scan_time,
                                    dev_bought_tokens, dev_bought_sol, dev_bought_percentage,
                                    dev_bought_curve_percentage, dev_created_tokens,
                                    same_name_count, same_website_count, same_telegram_count,
                                    same_twitter_count, bundle_count, bundle_supply_percentage,
                                    bundle_curve_percentage, bundle_sol, notable_bundle_count,
                                    notable_bundle_supply_percentage, notable_bundle_curve_percentage,
                                    notable_bundle_sol, sniper_activity_tokens,
                                    sniper_activity_percentage, sniper_activity_sol,
                                    created_time, traders_count, traders_last_swap,
                                    holders_total, holders_top10_percentage,
                                    holders_top25_percentage, holders_top50_percentage,
                                    dev_holds, dev_sold_times, dev_sold_sol,
                                    dev_sold_percentage
                                ) VALUES (
                                    ?, CURRENT_TIMESTAMP,
                                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                                )
                            ''', (
                                address,
                                syrax_data.get('dev_bought', {}).get('tokens', 0),  # Pataisyta struktūra
                                syrax_data.get('dev_bought', {}).get('sol', 0),
                                syrax_data.get('dev_bought', {}).get('percentage', 0),
                                syrax_data.get('dev_bought', {}).get('curve_percentage', 0),
                                syrax_data.get('dev_created_tokens'),
                                syrax_data.get('same_name_count'),
                                syrax_data.get('same_website_count'),
                                syrax_data.get('same_telegram_count'),
                                syrax_data.get('same_twitter_count'),
                                syrax_data.get('bundle', {}).get('count'),  # Pataisyta struktūra
                                syrax_data.get('bundle', {}).get('supply_percentage'),
                                syrax_data.get('bundle', {}).get('curve_percentage'),
                                syrax_data.get('bundle', {}).get('sol'),
                                syrax_data.get('notable_bundle', {}).get('count'),  # Pataisyta struktūra
                                syrax_data.get('notable_bundle', {}).get('supply_percentage'),
                                syrax_data.get('notable_bundle', {}).get('curve_percentage'),
                                syrax_data.get('notable_bundle', {}).get('sol'),
                                syrax_data.get('sniper_activity', {}).get('tokens', 0),  # Pataisyta struktūra
                                syrax_data.get('sniper_activity', {}).get('percentage'),
                                syrax_data.get('sniper_activity', {}).get('sol'),
                                syrax_data.get('created_time'),
                                syrax_data.get('traders', {}).get('count'),  # Pataisyta struktūra
                                syrax_data.get('traders', {}).get('last_swap'),
                                syrax_data.get('holders', {}).get('total'),  # Pataisyta struktūra
                                syrax_data.get('holders', {}).get('top10_percentage'),
                                syrax_data.get('holders', {}).get('top25_percentage'),
                                syrax_data.get('holders', {}).get('top50_percentage'),
                                syrax_data.get('dev_holds'),  # Tiesiogiai iš dev_holds
                                syrax_data.get('dev_sold', {}).get('times'),  # Iš dev_sold nested objekto
                                syrax_data.get('dev_sold', {}).get('sol'),  # Iš dev_sold nested objekto
                                syrax_data.get('dev_sold', {}).get('percentage')
                            ))
                            
                        except Exception as e:
                           
                            raise
                    
                    # Proficy Price duomenų įrašymas
                    if proficy_data:
                        
                        self.cursor.execute('''
                            INSERT INTO proficy_price_data (
                                token_address, scan_time,
                                price_change_5m, volume_5m, bs_ratio_5m,
                                price_change_1h, volume_1h, bs_ratio_1h
                            ) VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
                        ''', (
                            address,
                            proficy_data.get('5m', {}).get('price_change'),
                            proficy_data.get('5m', {}).get('volume'),
                            proficy_data.get('5m', {}).get('bs_ratio'),
                            proficy_data.get('1h', {}).get('price_change'),
                            proficy_data.get('1h', {}).get('volume'),
                            proficy_data.get('1h', {}).get('bs_ratio')
                        ))
                        
                except Exception as e:
                    
                    raise

            if not is_new_token:  # Kai tai UPDATE
                initial_mc, multiplier = self.calculate_multiplier(address, current_mc)
                
                if initial_mc > 0 and multiplier > 0:
                    # Spausdiname info apie multiplier
                    print(f"\n{'='*50}")
                    print(f"Token: {address}")
                    print(f"Initial Market Cap: {initial_mc:,.2f} USD")
                    print(f"Current Market Cap: {current_mc:,.2f} USD")
                    print(f"Current Multiplier: {multiplier:.2f}x")
                    print(f"{'='*50}\n")
                    
                    # Jei pasiekė GEM_MULTIPLIER
                    # Patikriname ar šis token'as jau yra gem_tokens lentelėje
                    self.cursor.execute('''
                        SELECT token_address 
                        FROM gem_tokens 
                        WHERE token_address = ?
                    ''', (address,))
                    
                    already_gem = self.cursor.fetchone() is not None
                    
                    if not already_gem and multiplier >= float(Config.GEM_MULTIPLIER.replace('x', '')):  # Įrašome į gem_tokens TIK jei dar nėra IR multiplier >= 10
                        print(f"🌟 Token {address} has reached {multiplier:.2f}x and is now marked as GEM!")
                        
                        # TIK ČIA atnaujiname is_gem statusą, kai token'as tikrai tapo GEM
                        self.cursor.execute('''
                            UPDATE tokens 
                            SET is_gem = TRUE,
                            last_updated = CURRENT_TIMESTAMP
                            WHERE address = ?
                        ''', (address,))
                        
                        # Gauname pradinius duomenis
                        self.cursor.execute('''
                            SELECT * FROM soul_scanner_data 
                            WHERE token_address = ? 
                            ORDER BY scan_time ASC 
                            LIMIT 1
                        ''', (address,))
                        initial_soul_data = dict(self.cursor.fetchone())
                        
                        self.cursor.execute('''
                            SELECT * FROM syrax_scanner_data 
                            WHERE token_address = ? 
                            ORDER BY scan_time ASC 
                            LIMIT 1
                        ''', (address,))
                        initial_syrax_data = dict(self.cursor.fetchone())

                        self.cursor.execute('''
                            SELECT * FROM proficy_price_data 
                            WHERE token_address = ? 
                            ORDER BY scan_time ASC 
                            LIMIT 1
                        ''', (address,))
                        initial_proficy_data = dict(self.cursor.fetchone())
                        
                        # Įrašome į gem_tokens ML analizei
                        # gem_tokens INSERT užklausą pakeisti į:
                        self.cursor.execute('''
                            INSERT OR IGNORE INTO gem_tokens (
                                token_address,
                                -- Soul Scanner pradiniai duomenys
                                initial_name, initial_symbol, initial_market_cap, initial_ath_market_cap,
                                initial_liquidity_usd, initial_liquidity_sol, initial_mint_status,
                                initial_freeze_status, initial_lp_status, initial_dex_status_paid,
                                initial_dex_status_ads, initial_total_scans, initial_social_link_x,
                                initial_social_link_tg, initial_social_link_web,
                                
                                -- Syrax Scanner pradiniai duomenys
                                initial_dev_bought_tokens, initial_dev_bought_sol, initial_dev_bought_percentage,
                                initial_dev_bought_curve_percentage, initial_dev_created_tokens,
                                initial_same_name_count, initial_same_website_count, initial_same_telegram_count,
                                initial_same_twitter_count, initial_bundle_count, initial_bundle_supply_percentage,
                                initial_bundle_curve_percentage, initial_bundle_sol, initial_notable_bundle_count,
                                initial_notable_bundle_supply_percentage, initial_notable_bundle_curve_percentage,
                                initial_notable_bundle_sol, initial_sniper_activity_tokens,
                                initial_sniper_activity_percentage, initial_sniper_activity_sol,
                                initial_created_time, initial_traders_count, initial_traders_last_swap,
                                initial_holders_total, initial_holders_top10_percentage,
                                initial_holders_top25_percentage, initial_holders_top50_percentage,
                                initial_dev_holds, initial_dev_sold_times, initial_dev_sold_sol,
                                initial_dev_sold_percentage,
                                
                                -- Proficy pradiniai duomenys
                                initial_price_change_5m, initial_volume_5m, initial_bs_ratio_5m,
                                initial_price_change_1h, initial_volume_1h, initial_bs_ratio_1h,
                                
                                -- ML rezultatai
                                similarity_score, confidence_level, recommendation, avg_z_score, is_passed, discovery_time
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                      ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                      100, 100, 'CONFIRMED GEM', 0.0, True, CURRENT_TIMESTAMP)
                        ''', (
                            address,
                            # Soul Scanner duomenys
                            initial_soul_data.get('name'),
                            initial_soul_data.get('symbol'),
                            initial_soul_data.get('market_cap', 0),
                            initial_soul_data.get('ath_market_cap', 0),
                            initial_soul_data.get('liquidity_usd', 0),
                            initial_soul_data.get('liquidity_sol', 0),
                            initial_soul_data.get('mint_status', 0),
                            initial_soul_data.get('freeze_status', 0),
                            initial_soul_data.get('lp_status', 0),
                            initial_soul_data.get('dex_status_paid', 0),
                            initial_soul_data.get('dex_status_ads', 0),
                            initial_soul_data.get('total_scans', 0),
                            initial_soul_data.get('social_link_x', ''),
                            initial_soul_data.get('social_link_tg', ''),
                            initial_soul_data.get('social_link_web', ''),
                            
                            # Syrax Scanner duomenys
                            initial_syrax_data.get('dev_bought', {}).get('tokens', 0),
                            initial_syrax_data.get('dev_bought', {}).get('sol', 0),
                            initial_syrax_data.get('dev_bought', {}).get('percentage', 0),
                            initial_syrax_data.get('dev_bought', {}).get('curve_percentage', 0),
                            initial_syrax_data.get('dev_created_tokens', 0),
                            initial_syrax_data.get('same_name_count', 0),
                            initial_syrax_data.get('same_website_count', 0),
                            initial_syrax_data.get('same_telegram_count', 0),
                            initial_syrax_data.get('same_twitter_count', 0),
                            initial_syrax_data.get('bundle_count', 0),
                            initial_syrax_data.get('bundle_supply_percentage', 0),
                            initial_syrax_data.get('bundle_curve_percentage', 0),
                            initial_syrax_data.get('bundle_sol', 0),
                            initial_syrax_data.get('notable_bundle_count', 0),
                            initial_syrax_data.get('notable_bundle_supply_percentage', 0),
                            initial_syrax_data.get('notable_bundle_curve_percentage', 0),
                            initial_syrax_data.get('notable_bundle_sol', 0),
                            initial_syrax_data.get('sniper_activity', {}).get('tokens', 0),
                            initial_syrax_data.get('sniper_activity_percentage', 0),
                            initial_syrax_data.get('sniper_activity_sol', 0),
                            initial_syrax_data.get('created_time', ''),
                            initial_syrax_data.get('traders_count', 0),
                            initial_syrax_data.get('traders_last_swap', ''),
                            initial_syrax_data.get('holders_total', 0),
                            initial_syrax_data.get('holders_top10_percentage', 0),
                            initial_syrax_data.get('holders_top25_percentage', 0),
                            initial_syrax_data.get('holders_top50_percentage', 0),
                            initial_syrax_data.get('dev_holds', 0),
                            initial_syrax_data.get('dev_sold_times', 0),
                            initial_syrax_data.get('dev_sold_sol', 0),
                            initial_syrax_data.get('dev_sold_percentage', 0),
                            
                            # Proficy duomenys
                            initial_proficy_data.get('price_change_5m', 0),
                            initial_proficy_data.get('volume_5m', 0),
                            initial_proficy_data.get('bs_ratio_5m', 0),
                            initial_proficy_data.get('price_change_1h', 0),
                            initial_proficy_data.get('volume_1h', 0),
                            
                            initial_proficy_data.get('bs_ratio_1h', 0)
                            
                                                    ))

            
            # LOGGER 7: Įsitikiname, kad viskas išsaugota
            
            try:
                self.conn.commit()
                print(f"[DEBUG] All changes committed successfully")

                                
                # LOGGER 8: Galutinis patikrinimas
                self.cursor.execute("SELECT * FROM tokens WHERE address = ?", (address,))
                final_check = self.cursor.fetchone()
                print(f"[DEBUG] Final check - token in database: {bool(final_check)}")
                
                if final_check:
                    print(f"[DEBUG] Token status - is_gem: {final_check['is_gem']}, total_scans: {final_check['total_scans']}")
                
                # Patikriname ar įrašyti scanner'ių duomenys
                self.cursor.execute("SELECT COUNT(*) FROM soul_scanner_data WHERE token_address = ?", (address,))
                soul_count = self.cursor.fetchone()[0]
                self.cursor.execute("SELECT COUNT(*) FROM syrax_scanner_data WHERE token_address = ?", (address,))
                syrax_count = self.cursor.fetchone()[0]
                self.cursor.execute("SELECT COUNT(*) FROM proficy_price_data WHERE token_address = ?", (address,))
                proficy_count = self.cursor.fetchone()[0]
                
                print(f"[DEBUG] Scanner data records:")
                print(f"- Soul Scanner records: {soul_count}")
                print(f"- Syrax Scanner records: {syrax_count}")
                print(f"- Proficy Price records: {proficy_count}")
                
                return True

            except Exception as e:
                
                self.conn.rollback()
                raise

        except Exception as e:
            logger.error(f"Error saving token data: {e}")
            print(f"[ERROR] Global error in save_token_data: {str(e)}")
            print(f"[ERROR] Error type: {type(e).__name__}")
            self.conn.rollback()
            return False

    def delete_token(self, token_address: str):
        try:
            self.cursor.execute("""
                DELETE FROM proficy_price_data WHERE token_address = %s;
                DELETE FROM syrax_scanner_data WHERE token_address = %s;
                DELETE FROM soul_scanner_data WHERE token_address = %s;
                DELETE FROM gem_tokens WHERE token_address = %s;
                DELETE FROM tokens WHERE address = %s;
            """, (token_address, token_address, token_address, token_address, token_address))
            self.connection.commit()
            print(f"Successfully deleted token {token_address} and all related data")
        except Exception as e:
            self.connection.rollback()
            print(f"Error deleting token: {e}")
    
    def load_gem_tokens(self) -> List[Dict]:
        """Užkrauna visus GEM token'us su jų pradiniais duomenimis ML analizei"""
        try:
            print("\n=== Running GEM Data Diagnostics ===")
            self.diagnose_gem_data()  # Pridedame diagnostiką
            print("\n=== LOADING GEM TOKENS FROM DATABASE ===")

           
            self.cursor.execute('''
            SELECT 
                t.address,
                t.first_seen,
                -- Soul Scanner pradiniai duomenys
                s.name,
                s.symbol,
                s.market_cap,
                s.ath_market_cap,
                s.liquidity_usd,
                s.liquidity_sol,
                s.mint_status,
                s.freeze_status,
                s.lp_status,
                s.dex_status_paid,
                s.dex_status_ads,
                s.total_scans,
                s.social_link_x,
                s.social_link_tg,
                s.social_link_web,
                -- Syrax Scanner pradiniai duomenys
                sy.dev_bought_tokens,
                sy.dev_bought_sol,
                sy.dev_bought_percentage,
                sy.dev_bought_curve_percentage,
                sy.dev_created_tokens,
                sy.same_name_count,
                sy.same_website_count,
                sy.same_telegram_count,
                sy.same_twitter_count,
                sy.bundle_count,
                sy.bundle_supply_percentage,
                sy.bundle_curve_percentage,
                sy.bundle_sol,
                sy.notable_bundle_count,
                sy.notable_bundle_supply_percentage,
                sy.notable_bundle_curve_percentage,
                sy.notable_bundle_sol,
                sy.sniper_activity_tokens,
                sy.sniper_activity_percentage,
                sy.sniper_activity_sol,
                sy.created_time,
                sy.traders_count,
                sy.traders_last_swap,
                sy.holders_total,
                sy.holders_top10_percentage,
                sy.holders_top25_percentage,
                sy.holders_top50_percentage,
                sy.dev_holds,
                sy.dev_sold_times,
                sy.dev_sold_sol,
                sy.dev_sold_percentage,
                -- Proficy pradiniai duomenys
                p.price_change_5m,
                p.volume_5m,
                p.bs_ratio_5m,
                p.price_change_1h,
                p.volume_1h,
                p.bs_ratio_1h,
                -- GEM analizės rezultatai
                g.similarity_score,
                g.confidence_level,
                g.recommendation,
                g.avg_z_score,
                g.is_passed,
                g.discovery_time
            FROM tokens t
            JOIN gem_tokens g ON t.address = g.token_address
            JOIN soul_scanner_data s ON t.address = s.token_address
            JOIN syrax_scanner_data sy ON t.address = sy.token_address
            JOIN proficy_price_data p ON t.address = p.token_address
            WHERE t.is_gem = TRUE
            AND s.scan_time = (
                SELECT MIN(scan_time) 
                FROM soul_scanner_data 
                WHERE token_address = t.address
            )
            AND sy.scan_time = (
                SELECT MIN(scan_time) 
                FROM syrax_scanner_data 
                WHERE token_address = t.address
            )
            AND p.scan_time = (
                SELECT MIN(scan_time) 
                FROM proficy_price_data 
                WHERE token_address = t.address
            )
            ORDER BY g.discovery_time DESC
            ''')
            rows = self.cursor.fetchall()
            tokens = [dict(row) for row in rows]
            
            print(f"\nLoaded {len(tokens)} GEM tokens")
        
            # Išsami informacija apie kiekvieną token'ą
            for i, token in enumerate(tokens, 1):
                print(f"\n=== GEM Token #{i} ===")
                print(f"Address: {token.get('address')}")
                print(f"First Seen: {token.get('first_seen')}")
                
                print("\nSoul Scanner Data:")
                print(f"Name: {token.get('name')}")
                print(f"Symbol: {token.get('symbol')}")
                print(f"Market Cap: {token.get('market_cap')}")
                print(f"ATH Market Cap: {token.get('ath_market_cap')}")
                print(f"Liquidity USD: {token.get('liquidity_usd')}")
                print(f"Liquidity SOL: {token.get('liquidity_sol')}")
                print(f"Mint Status: {token.get('mint_status')}")
                print(f"Freeze Status: {token.get('freeze_status')}")
                print(f"LP Status: {token.get('lp_status')}")
                print(f"DEX Status Paid: {token.get('dex_status_paid')}")
                print(f"DEX Status Ads: {token.get('dex_status_ads')}")
                print(f"Total Scans: {token.get('total_scans')}")
                print(f"Social X: {token.get('social_link_x')}")
                print(f"Social TG: {token.get('social_link_tg')}")
                print(f"Social Web: {token.get('social_link_web')}")
                
                print("\nSyrax Scanner Data:")
                print("Dev Bought:")
                print(f"- Tokens: {token.get('dev_bought_tokens')}")
                print(f"- SOL: {token.get('dev_bought_sol')}")
                print(f"- Percentage: {token.get('dev_bought_percentage')}")
                print(f"- Curve %: {token.get('dev_bought_curve_percentage')}")
                print(f"Dev Created Tokens: {token.get('dev_created_tokens')}")
                print("Similar Tokens:")
                print(f"- Name: {token.get('same_name_count')}")
                print(f"- Website: {token.get('same_website_count')}")
                print(f"- Telegram: {token.get('same_telegram_count')}")
                print(f"- Twitter: {token.get('same_twitter_count')}")
                print("Bundle Info:")
                print(f"- Count: {token.get('bundle_count')}")
                print(f"- Supply %: {token.get('bundle_supply_percentage')}")
                print(f"- Curve %: {token.get('bundle_curve_percentage')}")
                print(f"- SOL: {token.get('bundle_sol')}")
                print("Notable Bundle:")
                print(f"- Count: {token.get('notable_bundle_count')}")
                print(f"- Supply %: {token.get('notable_bundle_supply_percentage')}")
                print(f"- Curve %: {token.get('notable_bundle_curve_percentage')}")
                print(f"- SOL: {token.get('notable_bundle_sol')}")
                print("Sniper Activity:")
                print(f"- Tokens: {token.get('sniper_activity_tokens')}")
                print(f"- Percentage: {token.get('sniper_activity_percentage')}")
                print(f"- SOL: {token.get('sniper_activity_sol')}")
                print(f"Created Time: {token.get('created_time')}")
                print(f"Traders Count: {token.get('traders_count')}")
                print(f"Last Swap: {token.get('traders_last_swap')}")
                print("Holders:")
                print(f"- Total: {token.get('holders_total')}")
                print(f"- Top 10%: {token.get('holders_top10_percentage')}")
                print(f"- Top 25%: {token.get('holders_top25_percentage')}")
                print(f"- Top 50%: {token.get('holders_top50_percentage')}")
                print("Dev Info:")
                print(f"- Holds: {token.get('dev_holds')}")
                print(f"- Sold Times: {token.get('dev_sold_times')}")
                print(f"- Sold SOL: {token.get('dev_sold_sol')}")
                print(f"- Sold %: {token.get('dev_sold_percentage')}")
                
                print("\nProficy Data:")
                print("5min:")
                print(f"- Price Change: {token.get('price_change_5m')}")
                print(f"- Volume: {token.get('volume_5m')}")
                print(f"- B/S Ratio: {token.get('bs_ratio_5m')}")
                print("1hour:")
                print(f"- Price Change: {token.get('price_change_1h')}")
                print(f"- Volume: {token.get('volume_1h')}")
                print(f"- B/S Ratio: {token.get('bs_ratio_1h')}")

                print("\nAnalysis Results:")
                print(f"Similarity Score: {token.get('similarity_score')}")
                print(f"Confidence Level: {token.get('confidence_level')}")
                print(f"Recommendation: {token.get('recommendation')}")
                print(f"Avg Z-Score: {token.get('avg_z_score')}")
                print(f"Is Passed: {token.get('is_passed')}")
                print(f"Discovery Time: {token.get('discovery_time')}")
            
            # Palikta originali NULL reikšmių patikra
            print("\n=== Checking for NULL values ===")
            for token in tokens:
                for key, value in token.items():
                    if value is None:
                        print(f"NULL value found in {token['address']} for field: {key}")
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error loading GEM tokens: {e}")
            print(f"\nERROR loading GEM tokens: {str(e)}")
            return []
    
        
    def save_ml_intervals(self, intervals: Dict):
        """Išsaugo ML intervalų duomenis"""
        try:
            for feature, values in intervals.items():
                self.cursor.execute('''
                INSERT OR REPLACE INTO ml_intervals (
                    feature_name, min_value, max_value, mean_value, std_value
                ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    feature,
                    values.get('min'),
                    values.get('max'),
                    values.get('mean'),
                    values.get('std')
                ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving ML intervals: {e}")
            self.conn.rollback()
            return False

    
    def load_ml_intervals(self) -> Dict:
        """Užkrauna ML intervalų duomenis"""
        try:
            self.cursor.execute('SELECT * FROM ml_intervals')
            rows = self.cursor.fetchall()
            return {row['feature_name']: {
                'min': row['min_value'],
                'max': row['max_value'],
                'mean': row['mean_value'],
                'std': row['std_value']
            } for row in rows}
        except Exception as e:
            logger.error(f"Error loading ML intervals: {e}")
            return {}

    def close(self):
        """Uždaro duomenų bazės prisijungimą"""
        try:
            self.conn.close()
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

    def display_database_stats(self):
        """Parodo išsamią duomenų bazės statistiką"""
        try:
            print("\n=== DATABASE CONTENT ===")
            
            # Pilna užklausa su visais duomenimis
            self.cursor.execute("""
                WITH LatestData AS (
                    SELECT 
                        t.address,
                        t.first_seen,
                        t.last_updated,
                        t.is_gem,
                        t.total_scans,
                        s.name,
                        s.symbol,
                        s.market_cap,
                        s.ath_market_cap,
                        s.liquidity_usd,
                        s.liquidity_sol,
                        s.mint_status,
                        s.freeze_status,
                        s.lp_status,
                        s.dex_status_paid,
                        s.dex_status_ads,
                        s.social_link_x,
                        s.social_link_tg,
                        s.social_link_web,
                        sy.dev_bought_tokens,
                        sy.dev_bought_sol,
                        sy.dev_bought_percentage,
                        sy.dev_bought_curve_percentage,
                        sy.dev_created_tokens,
                        sy.same_name_count,
                        sy.same_website_count,
                        sy.same_telegram_count,
                        sy.same_twitter_count,
                        sy.bundle_count,
                        sy.bundle_supply_percentage,
                        sy.bundle_curve_percentage,
                        sy.bundle_sol,
                        sy.notable_bundle_count,
                        sy.notable_bundle_supply_percentage,
                        sy.notable_bundle_curve_percentage,
                        sy.notable_bundle_sol,
                        sy.sniper_activity_tokens,
                        sy.sniper_activity_percentage,
                        sy.sniper_activity_sol,
                        sy.holders_total,
                        sy.holders_top10_percentage,
                        sy.holders_top25_percentage,
                        sy.holders_top50_percentage,
                        sy.dev_holds,
                        sy.dev_sold_times,
                        sy.dev_sold_sol,
                        sy.dev_sold_percentage,
                        p.price_change_5m,
                        p.volume_5m,
                        p.bs_ratio_5m,
                        p.price_change_1h,
                        p.volume_1h,
                        p.bs_ratio_1h,
                        ROW_NUMBER() OVER (PARTITION BY t.address ORDER BY t.last_updated DESC) as rn
                    FROM tokens t
                    LEFT JOIN soul_scanner_data s ON t.address = s.token_address
                    LEFT JOIN syrax_scanner_data sy ON t.address = sy.token_address
                    LEFT JOIN proficy_price_data p ON t.address = p.token_address
                )
                SELECT * FROM LatestData WHERE rn = 1
                ORDER BY first_seen DESC
            """)
            
            columns = [description[0] for description in self.cursor.description]
            tokens = []
            for row in self.cursor.fetchall():
                token_dict = {}
                for i, column in enumerate(columns):
                    if column != 'rn':
                        token_dict[column] = row[i]
                tokens.append(token_dict)

            for token in tokens:
                print("\n==================== TOKEN INFO ====================")
                print("Basic Info:")
                print(f"Address: {token['address']}")
                print(f"First Seen: {token['first_seen']}")
                print(f"Last Updated: {token['last_updated']}")
                print(f"Is GEM: {'Yes' if token['is_gem'] else 'No'}")
                print(f"Total Scans: {token['total_scans']}")
                
                print("\nSoul Scanner Data:")
                print(f"Name: {token['name']}")
                print(f"Symbol: {token['symbol']}")
                print(f"Market Cap: ${token['market_cap']:,.2f}" if token['market_cap'] else "Market Cap: N/A")
                print(f"ATH Market Cap: ${token['ath_market_cap']:,.2f}" if token['ath_market_cap'] else "ATH Market Cap: N/A")
                print(f"Liquidity USD: ${token['liquidity_usd']:,.2f}" if token['liquidity_usd'] else "Liquidity USD: N/A")
                print(f"Liquidity SOL: {token['liquidity_sol']}" if token['liquidity_sol'] else "Liquidity SOL: N/A")
                print(f"Mint Status: {token['mint_status']}")
                print(f"Freeze Status: {token['freeze_status']}")
                print(f"LP Status: {token['lp_status']}")
                print(f"DEX Status Paid: {token['dex_status_paid']}")
                print(f"DEX Status Ads: {token['dex_status_ads']}")
                print(f"Social Links:")
                print(f"  X: {token['social_link_x']}")
                print(f"  TG: {token['social_link_tg']}")
                print(f"  WEB: {token['social_link_web']}")
                
                print("\nSyrax Scanner Data:")
                print(f"Dev Bought:")
                if token['dev_bought_tokens']:
                    try:
                        print(f"  Tokens: {float(token['dev_bought_tokens']):,.2f}")
                    except (ValueError, TypeError):
                        print(f"  Tokens: {token['dev_bought_tokens']}")
                else:
                    print("  Tokens: N/A")
                print(f"  SOL: {token['dev_bought_sol']}")
                print(f"  Percentage: {token['dev_bought_percentage']}%")
                print(f"  Curve Percentage: {token['dev_bought_curve_percentage']}%")
                print(f"Dev Created Tokens: {token['dev_created_tokens']}")
                print(f"Similar Tokens:")
                print(f"  Same Name: {token['same_name_count']}")
                print(f"  Same Website: {token['same_website_count']}")
                print(f"  Same Telegram: {token['same_telegram_count']}")
                print(f"  Same Twitter: {token['same_twitter_count']}")
                print(f"Bundle Info:")
                print(f"  Count: {token['bundle_count']}")
                print(f"  Supply %: {token['bundle_supply_percentage']}")
                print(f"  Curve %: {token['bundle_curve_percentage']}")
                print(f"  SOL: {token['bundle_sol']}")
                print(f"Notable Bundle Info:")
                print(f"  Count: {token['notable_bundle_count']}")
                print(f"  Supply %: {token['notable_bundle_supply_percentage']}")
                print(f"  Curve %: {token['notable_bundle_curve_percentage']}")
                print(f"  SOL: {token['notable_bundle_sol']}")
                print(f"Sniper Activity:")
                print(f"  Tokens: {token['sniper_activity_tokens']:,.2f}" if token['sniper_activity_tokens'] else "  Tokens: N/A")
                print(f"  Percentage: {token['sniper_activity_percentage']}")
                print(f"  SOL: {token['sniper_activity_sol']}")
                print(f"Holders Info:")
                print(f"  Total: {token['holders_total']}")
                print(f"  Top 10%: {token['holders_top10_percentage']}")
                print(f"  Top 25%: {token['holders_top25_percentage']}")
                print(f"  Top 50%: {token['holders_top50_percentage']}")
                print(f"Dev Info:")
                print(f"  Holds: {token['dev_holds']}")
                print(f"  Sold Times: {token['dev_sold_times']}")
                print(f"  Sold SOL: {token['dev_sold_sol']}")
                print(f"  Sold Percentage: {token['dev_sold_percentage']}")
                
                print("\nProficy Price Data:")
                print(f"5min:")
                print(f"  Price Change: {token['price_change_5m']}")
                print(f"  Volume: ${token['volume_5m']:,.2f}" if token['volume_5m'] else "  Volume: N/A")
                print(f"  B/S Ratio: {token['bs_ratio_5m']}")
                print(f"1hour:")
                print(f"  Price Change: {token['price_change_1h']}")
                print(f"  Volume: ${token['volume_1h']:,.2f}" if token['volume_1h'] else "  Volume: N/A")
                print(f"  B/S Ratio: {token['bs_ratio_1h']}")

            print("\n=== SUMMARY ===")
            print(f"Total Tokens: {len(tokens)}")
            self.cursor.execute("SELECT COUNT(*) FROM tokens WHERE is_gem = TRUE")
            gem_count = self.cursor.fetchone()[0]
            print(f"Total GEMs: {gem_count}")
            
            print("\n================================================")
            
        except Exception as e:
            logger.error(f"Error displaying database stats: {str(e)}")
            print(f"Database Error: {str(e)}")

    def diagnose_gem_data(self):
        """Diagnostika GEM duomenų"""
        try:
            print("\n=== GEM Data Diagnostics ===")
            
            # Tikriname tokens lentelę
            self.cursor.execute("""
                SELECT COUNT(*) as total_tokens,
                       SUM(CASE WHEN is_gem = TRUE THEN 1 ELSE 0 END) as gem_tokens
                FROM tokens
            """)
            token_counts = dict(self.cursor.fetchone())
            print(f"\nTokens table:")
            print(f"Total tokens: {token_counts['total_tokens']}")
            print(f"GEM tokens: {token_counts['gem_tokens']}")
            
            # Tikriname soul_scanner_data
            self.cursor.execute("""
                SELECT COUNT(DISTINCT token_address) as tokens,
                       COUNT(*) as total_records
                FROM soul_scanner_data
                WHERE token_address IN (SELECT address FROM tokens WHERE is_gem = TRUE)
            """)
            soul_counts = dict(self.cursor.fetchone())
            print(f"\nSoul Scanner Data:")
            print(f"Unique GEM tokens: {soul_counts['tokens']}")
            print(f"Total records: {soul_counts['total_records']}")
            
            # Tikriname syrax_scanner_data
            self.cursor.execute("""
                SELECT COUNT(DISTINCT token_address) as tokens,
                       COUNT(*) as total_records
                FROM syrax_scanner_data
                WHERE token_address IN (SELECT address FROM tokens WHERE is_gem = TRUE)
            """)
            syrax_counts = dict(self.cursor.fetchone())
            print(f"\nSyrax Scanner Data:")
            print(f"Unique GEM tokens: {syrax_counts['tokens']}")
            print(f"Total records: {syrax_counts['total_records']}")
            
            # Tikriname proficy_price_data
            self.cursor.execute("""
                SELECT COUNT(DISTINCT token_address) as tokens,
                       COUNT(*) as total_records
                FROM proficy_price_data
                WHERE token_address IN (SELECT address FROM tokens WHERE is_gem = TRUE)
            """)
            proficy_counts = dict(self.cursor.fetchone())
            print(f"\nProficy Price Data:")
            print(f"Unique GEM tokens: {proficy_counts['tokens']}")
            print(f"Total records: {proficy_counts['total_records']}")
            
            # Tikriname bendrus įrašus
            self.cursor.execute("""
                SELECT COUNT(DISTINCT t.address)
                FROM tokens t
                JOIN soul_scanner_data s ON t.address = s.token_address
                JOIN syrax_scanner_data sy ON t.address = sy.token_address
                JOIN proficy_price_data p ON t.address = p.token_address
                WHERE t.is_gem = TRUE
            """)
            common_tokens = self.cursor.fetchone()[0]
            print(f"\nTokens with data in ALL scanners: {common_tokens}")
            
        except Exception as e:
            print(f"Error during diagnostics: {str(e)}")

def initialize_database():
    """Inicializuoja duomenų bazę"""
    conn = sqlite3.connect('token_monitor.db')
    c = conn.cursor()
    
    # Pagrindinė tokens lentelė
    c.execute('''
    CREATE TABLE IF NOT EXISTS tokens (
        address TEXT PRIMARY KEY,
        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_gem BOOLEAN DEFAULT FALSE,
        total_scans INTEGER DEFAULT 1
    )''')

    # Soul Scanner duomenys
    c.execute('''
    CREATE TABLE IF NOT EXISTS soul_scanner_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        token_address TEXT NOT NULL,
        name TEXT,
        symbol TEXT,
        market_cap REAL,
        ath_market_cap REAL,
        liquidity_usd REAL,
        liquidity_sol REAL,
        mint_status BOOLEAN,
        freeze_status BOOLEAN,
        lp_status BOOLEAN,
        dex_status_paid BOOLEAN,
        dex_status_ads BOOLEAN,
        total_scans INTEGER,
        social_link_x TEXT,
        social_link_tg TEXT,
        social_link_web TEXT,
        scan_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (token_address) REFERENCES tokens(address)
    )''')

    # Syrax Scanner duomenys
    c.execute('''
    CREATE TABLE IF NOT EXISTS syrax_scanner_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        token_address TEXT NOT NULL,
        dev_bought_tokens REAL,
        dev_bought_sol REAL,
        dev_bought_percentage REAL,
        dev_bought_curve_percentage REAL,
        dev_created_tokens INTEGER,
        same_name_count INTEGER,
        same_website_count INTEGER,
        same_telegram_count INTEGER,
        same_twitter_count INTEGER,
        bundle_count INTEGER,
        bundle_supply_percentage REAL,
        bundle_curve_percentage REAL,
        bundle_sol REAL,
        notable_bundle_count INTEGER,
        notable_bundle_supply_percentage REAL,
        notable_bundle_curve_percentage REAL,
        notable_bundle_sol REAL,
        sniper_activity_tokens REAL,
        sniper_activity_percentage REAL,
        sniper_activity_sol REAL,
        created_time TIMESTAMP,
        traders_count INTEGER,
        traders_last_swap TEXT,
        holders_total INTEGER,
        holders_top10_percentage REAL,
        holders_top25_percentage REAL,
        holders_top50_percentage REAL,
        dev_holds INTEGER,
        dev_sold_times INTEGER,
        dev_sold_sol REAL,
        dev_sold_percentage REAL,
        scan_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (token_address) REFERENCES tokens(address)
    )''')

    # Proficy Price duomenys
    c.execute('''
    CREATE TABLE IF NOT EXISTS proficy_price_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        token_address TEXT NOT NULL,
        price_change_5m REAL,
        volume_5m REAL,
        bs_ratio_5m TEXT,
        price_change_1h REAL,
        volume_1h REAL,
        bs_ratio_1h TEXT,
        scan_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (token_address) REFERENCES tokens(address)
    )''')

    # GEM Token duomenys
    c.execute('''
            CREATE TABLE IF NOT EXISTS gem_tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        token_address TEXT NOT NULL,
        
        -- Soul Scanner pradiniai duomenys
        initial_name TEXT,
        initial_symbol TEXT,
        initial_market_cap REAL,
        initial_ath_market_cap REAL,
        initial_liquidity_usd REAL,
        initial_liquidity_sol REAL,
        initial_mint_status BOOLEAN,
        initial_freeze_status BOOLEAN,
        initial_lp_status BOOLEAN,
        initial_dex_status_paid BOOLEAN,
        initial_dex_status_ads BOOLEAN,
        initial_total_scans INTEGER,
        initial_social_link_x TEXT,
        initial_social_link_tg TEXT,
        initial_social_link_web TEXT,
        
        -- Syrax Scanner pradiniai duomenys
        initial_dev_bought_tokens REAL,
        initial_dev_bought_sol REAL,
        initial_dev_bought_percentage REAL,
        initial_dev_bought_curve_percentage REAL,
        initial_dev_created_tokens INTEGER,
        initial_same_name_count INTEGER,
        initial_same_website_count INTEGER,
        initial_same_telegram_count INTEGER,
        initial_same_twitter_count INTEGER,
        initial_bundle_count INTEGER,
        initial_bundle_supply_percentage REAL,
        initial_bundle_curve_percentage REAL,
        initial_bundle_sol REAL,
        initial_notable_bundle_count INTEGER,
        initial_notable_bundle_supply_percentage REAL,
        initial_notable_bundle_curve_percentage REAL,
        initial_notable_bundle_sol REAL,
        initial_sniper_activity_tokens REAL,
        initial_sniper_activity_percentage REAL,
        initial_sniper_activity_sol REAL,
        initial_created_time TIMESTAMP,
        initial_traders_count INTEGER,
        initial_traders_last_swap TEXT,
        initial_holders_total INTEGER,
        initial_holders_top10_percentage REAL,
        initial_holders_top25_percentage REAL,
        initial_holders_top50_percentage REAL,
        initial_dev_holds INTEGER,
        initial_dev_sold_times INTEGER,
        initial_dev_sold_sol REAL,
        initial_dev_sold_percentage REAL,
        
        -- Proficy pradiniai duomenys
        initial_price_change_5m REAL,
        initial_volume_5m REAL,
        initial_bs_ratio_5m TEXT,
        initial_price_change_1h REAL,
        initial_volume_1h REAL,
        initial_bs_ratio_1h TEXT,
        
        -- ML analizės rezultatai
        similarity_score REAL,
        confidence_level REAL,
        recommendation TEXT,
        avg_z_score REAL,
        is_passed BOOLEAN,
        discovery_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (token_address) REFERENCES tokens(address)
    )''')

    # ML Modelio intervalai
    c.execute('''
    CREATE TABLE IF NOT EXISTS ml_intervals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feature_name TEXT NOT NULL,
        min_value REAL,
        max_value REAL,
        mean_value REAL,
        std_value REAL,
        last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # Token analizės rezultatai
    c.execute('''
    CREATE TABLE IF NOT EXISTS token_analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        token_address TEXT NOT NULL,
        status TEXT,
        stage TEXT,
        similarity_score REAL,
        confidence_level REAL,
        recommendation TEXT,
        analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (token_address) REFERENCES tokens(address)
    )''')

    # Data ir vartotojas
    c.execute('''
    CREATE TABLE IF NOT EXISTS bot_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TIMESTAMP,
        user_login TEXT,
        last_active TIMESTAMP
    )''')

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

if __name__ == "__main__":
    try:
        # Inicializuojame duomenų bazę
        initialize_database()
        
        # Run the bot
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
