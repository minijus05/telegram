import re
import logging

import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from typing import Dict, List
from telethon import TelegramClient, events
import asyncio
from datetime import datetime, timezone
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
    TELEGRAM_SOURCE_CHATS = ['@solearlytrending', '@botubotass']
    TELEGRAM_DEST_CHAT = '@smartas1'
    TELEGRAM_GEM_CHAT = '@potentialGems'
    
    # Scanner settings
    SCANNER_GROUP = '@skaneriss'
    SOUL_SCANNER_BOT = 6872314605
    SYRAX_SCANNER_BOT = 7488438206
    PROFICY_PRICE_BOT = 5457577145

    # ML settings
    MIN_GEMS_FOR_ANALYSIS = 10  # Minimalus GEM skaičius prieš pradedant analizę

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

    async def display_stats(self):
        """Parodo duomenų bazės statistiką"""
        self.db.display_database_stats()

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
            token_addresses = self._extract_token_addresses(message)
            
            if token_addresses:
                for address in token_addresses:
                    # Siunčiame į scanner grupę
                    original_message = await self.scanner_client.send_message(
                        Config.SCANNER_GROUP,
                        address
                    )
                    logger.info(f"Sent token to scanner group: {address}")
                    print(f"\nNaujas token adresas: {address}\n")

                    # Renkame scannerių duomenis
                    scanner_data = await self._collect_scanner_data(address, original_message)
                    
                    if scanner_data:
                        # Išsaugome token duomenis į DB
                        self.db.save_token_data(
                            address,
                            scanner_data['soul'],
                            scanner_data['syrax'],
                            scanner_data['proficy']
                        )
                        
                        # Jei tai "from" žinutė su "10x" - pridedame į GEM duomenų bazę
                        if "from" in event.message.text.lower() and "10x" in event.message.text:
                            self.gem_analyzer.add_gem_token(scanner_data)
                            logger.info(f"Added successful GEM to database: {address}")
                        
                        # Bandome analizuoti, bet tik jei yra pakankamai duomenų
                        analysis_result = self.gem_analyzer.analyze_token(scanner_data)
                        
                        if analysis_result['status'] == 'pending':
                            print(f"\n{analysis_result['message']}")
                        else:
                            await self._handle_analysis_results(analysis_result, scanner_data)

        except Exception as e:
            logger.error(f"Error handling message: {e}")

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
        """Formatuoja ir siunčia analizės rezultatus"""
        if analysis_result['status'] == 'success':
            message = self._format_analysis_message(analysis_result, scanner_data)
            
            # Jei yra geras GEM potencialas, siunčiame į GEM grupę
            if analysis_result['similarity_score'] >= 60:
                await self.telegram.send_message(
                    Config.TELEGRAM_GEM_CHAT,
                    message
                )
                logger.info(f"Sent potential GEM alert with {analysis_result['similarity_score']}% similarity")
            
            # Spausdiname rezultatus konsolėje
            print("\nAnalizės rezultatai:")
            print(message)

    def _format_analysis_message(self, analysis_result, scanner_data):
        """Formatuoja analizės rezultatų žinutę"""
        soul_data = scanner_data['soul']
        
        message = [
            f"🔍 TOKEN ANALYSIS REPORT {'🚀' if analysis_result['similarity_score'] >= 60 else '⚠️'}",
            f"\nToken: {soul_data['name']} (${soul_data['symbol']})",
            f"Contract: {soul_data['contract_address']}",
            f"\n🎯 GEM Potential Score: {analysis_result['similarity_score']:.1f}%",
            f"🎲 Confidence Level: {analysis_result['confidence_level']:.1f}%",
            f"📊 Recommendation: {analysis_result['recommendation']}",
            "\n🔑 Primary Parameters Status:",
        ]

        # Pridedame pirminių parametrų detales
        for param, details in analysis_result['primary_check']['details'].items():
            status = "✅" if details['in_range'] else "❌"
            message.append(f"{status} {param}: {details['value']:.2f}")

        # Pridedame pagrindinius metrix'us
        message.extend([
            f"\n💰 Market Metrics:",
            f"Market Cap: ${soul_data['market_cap']:,.2f}",
            f"Liquidity: ${soul_data['liquidity_usd']:,.2f}",
            f"Holders: {scanner_data['syrax']['holders_total']}",
            f"\n📈 Price Action (1H):",
            f"Change: {scanner_data['proficy']['price_1h']}%",
            f"Volume: ${scanner_data['proficy']['volume_1h']:,.2f}"
        ])

        return "\n".join(message)

    def _extract_token_addresses(self, message: str) -> List[str]:
        """Ištraukia token adresus iš žinutės"""
        matches = []
        
        try:
            # Ieškome token adreso URL'uose
            if "from" in message:  # Update žinutė
                # Ieškome soul_scanner_bot URL
                scanner_matches = re.findall(r'soul_scanner_bot/chart\?startapp=([A-Za-z0-9]{32,44})', message)
                if scanner_matches:
                    matches.extend(scanner_matches)
                    
            elif "New" in message:  # Nauja žinutė
                # Ieškome soul_sniper_bot ir soul_scanner_bot URL
                patterns = [
                    r'soul_sniper_bot\?start=\d+_([A-Za-z0-9]{32,44})',
                    r'soul_scanner_bot/chart\?startapp=([A-Za-z0-9]{32,44})'
                ]
                
                for pattern in patterns:
                    url_matches = re.findall(pattern, message)
                    if url_matches:
                        matches.extend(url_matches)
            
            # Pašaliname dublikatus ir filtruojame
            unique_matches = list(set(matches))
            valid_matches = [addr for addr in unique_matches if len(addr) >= 32 and len(addr) <= 44]
            
            if valid_matches:
                logger.info(f"[2025-01-31 13:14:41] Found token address: {valid_matches[0]}")
            
            return valid_matches
            
        except Exception as e:
            logger.error(f"[2025-01-31 13:14:41] Error extracting token address: {e}")
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
                    elif len(line.strip()) > 30 and not any(x in line for x in ['https://', '🌊', '🔫', '📈', '🔗', '•', '┗', '┣']):
                        data['contract_address'] = line.strip().replace('`', '')
                    
                                                               
                    # Market Cap and ATH
                    elif 'MC:' in line:
                        # Market Cap gali būti K arba M
                        mc_k = re.search(r'\$(\d+\.?\d*)K', clean_line)  # Ieškome K
                        mc_m = re.search(r'\$(\d+\.?\d*)M', clean_line)  # Ieškome M
                        
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
            data = {}
            lines = message.split('\n')
            
            def convert_volume(volume_str: str) -> float:
                """Helper function to convert volume with K or M suffix"""
                volume_str = volume_str.replace('$', '')
                if 'M' in volume_str:
                    return float(volume_str.replace('M', '')) * 1000000
                elif 'K' in volume_str:
                    return float(volume_str.replace('K', '')) * 1000
                return float(volume_str)
            
            for line in lines:
                if 'Price' in line and 'Volume' in line and 'B/S' in line:
                    continue
                    
                if '5M:' in line:
                    parts = line.split()
                    data['5m'] = {
                        'price_change': float(parts[1].replace('%', '')),
                        'volume': convert_volume(parts[2]),
                        'bs_ratio': parts[3]
                    }
                    
                if '1H:' in line:
                    parts = line.split()
                    data['1h'] = {
                        'price_change': float(parts[1].replace('%', '')),
                        'volume': convert_volume(parts[2]),
                        'bs_ratio': parts[3]
                    }
                    
            return data
            
        except Exception as e:
            logger.error(f"Error parsing ProficyPrice message: {e}")
            return {}

    
class MLIntervalAnalyzer:
    """ML klasė pirminių intervalų nustatymui"""
    def __init__(self):
        self.primary_features = [
            'dev_created_tokens',
            'same_name_count',
            'same_website_count',
            'same_telegram_count',
            'same_twitter_count'
        ]
        self.scaler = MinMaxScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.intervals = {feature: {'min': float('inf'), 'max': float('-inf')} for feature in self.primary_features}
        
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
                value = float(gem.get('syrax', {}).get(feature, 0))
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
                # Statistinė analizė
                q1 = np.percentile(normal_values, 25)
                q3 = np.percentile(normal_values, 75)
                iqr = q3 - q1
                
                self.intervals[feature] = {
                    'min': max(0, q1 - 1.5 * iqr),
                    'max': q3 + 1.5 * iqr,
                    'mean': np.mean(normal_values),
                    'std': np.std(normal_values)
                }
        
        logger.info(f"ML intervalai atnaujinti sėkmingai su {len(successful_gems)} GEM'ais")
        return True
        
    def check_primary_parameters(self, token_data: Dict) -> Dict:
        """Tikrina ar token'o parametrai patenka į ML nustatytus intervalus"""
        results = {}
        syrax_data = token_data.get('syrax', {})
        
        for feature in self.primary_features:
            value = float(syrax_data.get(feature, 0))
            interval = self.intervals[feature]
            
            in_range = interval['min'] <= value <= interval['max']
            z_score = abs((value - interval['mean']) / interval['std']) if interval['std'] > 0 else float('inf')
            
            results[feature] = {
                'value': value,
                'in_range': in_range,
                'z_score': z_score,
                'interval': interval
            }
            
        # Bendras rezultatas
        all_in_range = all(result['in_range'] for result in results.values())
        avg_z_score = np.mean([result['z_score'] for result in results.values()])
        
        return {
            'passed': all_in_range,
            'avg_z_score': avg_z_score,
            'details': results
        }

class MLGEMAnalyzer:
    def __init__(self):
        self.interval_analyzer = MLIntervalAnalyzer()
        self.scaler = MinMaxScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.db = DatabaseManager()
        
        # Visi parametrai analizei
        self.features = {
            'soul': [
                'market_cap', 'ath_market_cap', 'liquidity_usd', 'liquidity_sol',
                'mint_status', 'freeze_status', 'dex_status_paid', 'dex_status_ads',
                'total_scans'
            ],
            'syrax': [
                'dev_bought_tokens', 'dev_bought_sol', 'dev_bought_percentage',
                'dev_bought_curve_percentage', 'bundle_count', 'bundle_supply_percentage',
                'bundle_curve_percentage', 'bundle_sol', 'notable_bundle_count',
                'notable_bundle_supply_percentage', 'notable_bundle_curve_percentage',
                'notable_bundle_sol', 'sniper_activity_tokens', 'sniper_activity_percentage',
                'sniper_activity_sol', 'traders', 'holders_total', 'holders_top10',
                'holders_top25', 'holders_top50', 'dev_holds', 'dev_sold_times',
                'dev_sold_sol', 'dev_sold_percentage'
            ],
            'proficy': [
                'price_5m', 'volume_5m', 'bs_ratio_5m',
                'price_1h', 'volume_1h', 'bs_ratio_1h'
            ]
        }
        
        self.gem_tokens = []
        self.load_gem_data()

    def load_gem_data(self):
        """Užkrauna išsaugotus GEM duomenis ir apmoko modelius"""
        try:
            # Užkrauname iš duomenų bazės
            self.gem_tokens = self.db.load_gem_tokens()
            if self.gem_tokens:
                # Apmokome intervalų analizatorių
                self.interval_analyzer.calculate_intervals(self.gem_tokens)
                # Apmokome pagrindinį ML modelį
                self._train_main_model()
        except Exception as e:
            logger.error(f"Error loading GEM data: {e}")

    def add_gem_token(self, token_data: Dict):
        """Prideda naują GEM ir atnaujina modelius"""
        self.gem_tokens.append(token_data)
        self.interval_analyzer.calculate_intervals(self.gem_tokens)
        self._train_main_model()
        
        # Išsaugome intervalus į DB
        self.db.save_ml_intervals(self.interval_analyzer.intervals)

    def _train_main_model(self):
        """Apmoko pagrindinį ML modelį su visais parametrais"""
        if not self.gem_tokens or len(self.gem_tokens) < 3:
            return False

        X = self._prepare_training_data()
        if len(X) > 0:
            X_scaled = self.scaler.fit_transform(X)
            self.isolation_forest.fit(X_scaled)
            return True
        return False

    def _prepare_training_data(self):
        """Paruošia duomenis ML modelio apmokymui"""
        data = []
        for token in self.gem_tokens:
            features = []
            # Ištraukiame visus parametrus iš kiekvieno scannerio
            for scanner, params in self.features.items():
                for param in params:
                    value = self._extract_feature_value(token, scanner, param)
                    features.append(value)
            data.append(features)
        return np.array(data)

    def _extract_feature_value(self, token, scanner, feature):
        """Ištraukia parametro reikšmę iš token duomenų"""
        try:
            value = token.get(scanner, {}).get(feature, 0)
            # Konvertuojame boolean į int
            if isinstance(value, bool):
                return int(value)
            # Ištraukiame skaičius iš string'ų (pvz., "2.2K" -> 2200)
            if isinstance(value, str):
                if 'K' in value:
                    return float(value.replace('K', '')) * 1000
                if 'M' in value:
                    return float(value.replace('M', '')) * 1000000
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def analyze_token(self, token_data: Dict) -> Dict:
        """Pilna token'o analizė"""
        # Patikriname ar turime pakankamai duomenų analizei
        if len(self.gem_tokens) < Config.MIN_GEMS_FOR_ANALYSIS:
            return {
                'status': 'pending',
                'message': f'Renkami duomenys. Reikia bent {Config.MIN_GEMS_FOR_ANALYSIS} GEMų analizei. Dabartinis kiekis: {len(self.gem_tokens)}',
                'collected_gems': len(self.gem_tokens)
            }

        # Toliau vykdome analizę tik jei turime pakankamai duomenų
        primary_check = self.interval_analyzer.check_primary_parameters(token_data)
        
        if not primary_check['passed']:
            return {
                'status': 'failed',
                'stage': 'primary',
                'score': 0,
                'details': primary_check['details'],
                'message': 'Token nepraėjo pirminės filtracijos'
            }

        # 2. Pilna ML analizė
        features = []
        feature_details = {}
        
        # Renkame visus parametrus
        for scanner, params in self.features.items():
            scanner_data = {}
            for param in params:
                value = self._extract_feature_value(token_data, scanner, param)
                features.append(value)
                scanner_data[param] = value
            feature_details[scanner] = scanner_data

        # Normalizuojame ir analizuojame
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        # Gauname anomalijos score
        anomaly_score = self.isolation_forest.score_samples(X_scaled)[0]
        similarity_score = (anomaly_score + 1) / 2 * 100  # Konvertuojame į procentus

        # 3. Rezultatų paruošimas
        result = {
            'status': 'success',
            'stage': 'full',
            'primary_check': primary_check,
            'similarity_score': similarity_score,
            'feature_analysis': feature_details,
            'recommendation': self._generate_recommendation(similarity_score, primary_check['avg_z_score']),
            'confidence_level': self._calculate_confidence(similarity_score, primary_check['avg_z_score'])
        }

        return result

    def _generate_recommendation(self, similarity_score: float, z_score: float) -> str:
        if similarity_score >= 80 and z_score < 1.5:
            return "STRONG GEM POTENTIAL"
        elif similarity_score >= 60 and z_score < 2:
            return "MODERATE GEM POTENTIAL"
        elif similarity_score >= 40:
            return "WEAK GEM POTENTIAL"
        return "NOT RECOMMENDED"

    def _calculate_confidence(self, similarity_score: float, z_score: float) -> float:
        # Skaičiuojame pasitikėjimo lygį
        confidence = (similarity_score / 100) * (1 / (1 + z_score))
        return min(max(confidence * 100, 0), 100)  # Konvertuojame į procentus

    def add_gem_token(self, token_data: Dict):
        """Prideda naują GEM ir atnaujina modelius"""
        self.gem_tokens.append(token_data)
        self.interval_analyzer.calculate_intervals(self.gem_tokens)
        self._train_main_model()
        self._save_gem_data()

    def _save_gem_data(self):
        """Išsaugo GEM duomenis į failą"""
        with open('gem_tokens.json', 'w') as f:
            json.dump(self.gem_tokens, f)
            
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

    def display_database_stats(self):
        """Parodo duomenų bazės statistiką"""
        try:
            # Tokens statistika
            self.cursor.execute("SELECT COUNT(*) FROM tokens")
            total_tokens = self.cursor.fetchone()[0]
            
            # GEM tokens statistika
            self.cursor.execute("SELECT COUNT(*) FROM tokens WHERE is_gem = TRUE")
            gem_tokens = self.cursor.fetchone()[0]
            
            # Paskutiniai 5 tokenai
            self.cursor.execute("""
            SELECT t.address, t.first_seen, s.name, s.symbol, s.market_cap
            FROM tokens t
            LEFT JOIN soul_scanner_data s ON t.address = s.token_address
            ORDER BY t.first_seen DESC
            LIMIT 5
            """)
            recent_tokens = self.cursor.fetchall()
            
            # Išvedame statistiką
            print("\n=== DATABASE STATISTICS ===")
            print(f"Total Tokens: {total_tokens}")
            print(f"Total GEMs: {gem_tokens}")
            
            print("\n=== LAST 5 TOKENS ===")
            for token in recent_tokens:
                print(f"\nAddress: {token[0]}")
                print(f"First seen: {token[1]}")
                print(f"Name: {token[2]}")
                print(f"Symbol: {token[3]}")
                print(f"Market Cap: ${token[4]:,.2f}" if token[4] else "Market Cap: N/A")
            
            print("\n=====================")
            
        except Exception as e:
            logger.error(f"Error displaying database stats: {e}")

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

        @monitor.telegram.on(events.NewMessage(chats=Config.TELEGRAM_SOURCE_CHATS))
        async def message_handler(event):
            await monitor.handle_new_message(event)

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

    def save_token_data(self, address: str, soul_data: Dict, syrax_data: Dict, proficy_data: Dict):
        """Išsaugo visus token duomenis"""
        try:
            # Įrašome pagrindinį token
            self.cursor.execute('''
                INSERT OR REPLACE INTO tokens (address, last_updated)
                VALUES (?, CURRENT_TIMESTAMP)
            ''', (address,))

            # Soul Scanner duomenys
            if soul_data:
                self.cursor.execute('''
                INSERT INTO soul_scanner_data (
                    token_address, name, symbol, market_cap, ath_market_cap,
                    liquidity_usd, liquidity_sol, mint_status, freeze_status,
                    lp_status, dex_status_paid, dex_status_ads, total_scans,
                    social_link_x, social_link_tg, social_link_web
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

            # Syrax Scanner duomenys
            if syrax_data:
                self.cursor.execute('''
                INSERT INTO syrax_scanner_data (
                    token_address, dev_bought_tokens, dev_bought_sol,
                    dev_bought_percentage, dev_bought_curve_percentage,
                    dev_created_tokens, same_name_count, same_website_count,
                    same_telegram_count, same_twitter_count, bundle_count,
                    bundle_supply_percentage, bundle_curve_percentage,
                    bundle_sol, notable_bundle_count,
                    notable_bundle_supply_percentage,
                    notable_bundle_curve_percentage, notable_bundle_sol,
                    sniper_activity_tokens, sniper_activity_percentage,
                    sniper_activity_sol, created_time, traders_count,
                    traders_last_swap, holders_total, holders_top10_percentage,
                    holders_top25_percentage, holders_top50_percentage,
                    dev_holds, dev_sold_times, dev_sold_sol, dev_sold_percentage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                         ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    address,
                    syrax_data.get('dev_bought', {}).get('tokens'),
                    syrax_data.get('dev_bought', {}).get('sol'),
                    syrax_data.get('dev_bought', {}).get('percentage'),
                    syrax_data.get('dev_bought', {}).get('curve_percentage'),
                    syrax_data.get('dev_created_tokens'),
                    syrax_data.get('same_name_count'),
                    syrax_data.get('same_website_count'),
                    syrax_data.get('same_telegram_count'),
                    syrax_data.get('same_twitter_count'),
                    syrax_data.get('bundle', {}).get('count'),
                    syrax_data.get('bundle', {}).get('supply_percentage'),
                    syrax_data.get('bundle', {}).get('curve_percentage'),
                    syrax_data.get('bundle', {}).get('sol'),
                    syrax_data.get('notable_bundle', {}).get('count'),
                    syrax_data.get('notable_bundle', {}).get('supply_percentage'),
                    syrax_data.get('notable_bundle', {}).get('curve_percentage'),
                    syrax_data.get('notable_bundle', {}).get('sol'),
                    syrax_data.get('sniper_activity', {}).get('tokens'),
                    syrax_data.get('sniper_activity', {}).get('percentage'),
                    syrax_data.get('sniper_activity', {}).get('sol'),
                    syrax_data.get('created_time'),
                    syrax_data.get('traders', {}).get('count'),
                    syrax_data.get('traders', {}).get('last_swap'),
                    syrax_data.get('holders', {}).get('total'),
                    syrax_data.get('holders', {}).get('top10_percentage'),
                    syrax_data.get('holders', {}).get('top25_percentage'),
                    syrax_data.get('holders', {}).get('top50_percentage'),
                    syrax_data.get('dev_holds'),
                    syrax_data.get('dev_sold', {}).get('times'),
                    syrax_data.get('dev_sold', {}).get('sol'),
                    syrax_data.get('dev_sold', {}).get('percentage')
                ))

            # Proficy Price duomenys
            if proficy_data:
                self.cursor.execute('''
                INSERT INTO proficy_price_data (
                    token_address, price_change_5m, volume_5m, bs_ratio_5m,
                    price_change_1h, volume_1h, bs_ratio_1h
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    address,
                    proficy_data.get('5m', {}).get('price_change'),
                    proficy_data.get('5m', {}).get('volume'),
                    proficy_data.get('5m', {}).get('bs_ratio'),
                    proficy_data.get('1h', {}).get('price_change'),
                    proficy_data.get('1h', {}).get('volume'),
                    proficy_data.get('1h', {}).get('bs_ratio')
                ))

            self.conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error saving token data: {e}")
            self.conn.rollback()
            return False

    def save_gem_analysis(self, address: str, analysis_result: Dict):
        """Išsaugo GEM analizės rezultatus"""
        try:
            self.cursor.execute('''
            INSERT INTO gem_tokens (
                token_address, similarity_score, confidence_level,
                recommendation, avg_z_score, is_passed
            ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                address,
                analysis_result.get('similarity_score'),
                analysis_result.get('confidence_level'),
                analysis_result.get('recommendation'),
                analysis_result.get('primary_check', {}).get('avg_z_score'),
                analysis_result.get('primary_check', {}).get('passed', False)
            ))

            # Atnaujiname token'ą kaip GEM
            self.cursor.execute('''
            UPDATE tokens SET is_gem = TRUE WHERE address = ?
            ''', (address,))

            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving GEM analysis: {e}")
            self.conn.rollback()
            return False

    def save_ml_intervals(self, intervals: Dict):
        """Išsaugo ML intervalų duomenis"""
        try:
            for feature, values in intervals.items():
                self.cursor.execute('''
                INSERT INTO ml_intervals (
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

    def load_gem_tokens(self) -> List[Dict]:
        """Užkrauna visus GEM token'us mokymui"""
        try:
            self.cursor.execute('''
            SELECT t.address, s.*, sy.*, p.*
            FROM tokens t
            JOIN soul_scanner_data s ON t.address = s.token_address
            JOIN syrax_scanner_data sy ON t.address = sy.token_address
            JOIN proficy_price_data p ON t.address = p.token_address
            WHERE t.is_gem = TRUE
            ''')
            rows = self.cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error loading GEM tokens: {e}")
            return []

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
        
        print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current User's Login: minijus05\n")
        
        # Įrašome bot'o paleidimo informaciją
        conn = sqlite3.connect('token_monitor.db')
        c = conn.cursor()
        c.execute('''
        INSERT INTO bot_info (start_time, user_login, last_active)
        VALUES (?, ?, ?)
        ''', (
            datetime.now(timezone.utc),
            "minijus05",
            datetime.now(timezone.utc)
        ))
        conn.commit()
        conn.close()
        
        
                
        # Run the bot
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
