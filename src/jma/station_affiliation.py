from common.text import normalize_comment

# Comment（元の所属文字列） → 英語 Legend ラベル
# ・地方区分: すべて "JMA"
# ・大学: すべて "University"
# ・東京都＋青森県＋静岡県: すべて "Prefectural Gov."
_COMMENT_LABEL_MAP_RAW: dict[str, str] = {
	'Hokkaido District  北海道地方': 'JMA',
	'Tohoku District  東北地方': 'JMA',
	'Kanto and Chubu Districts  関東・中部地方': 'JMA',
	'Kinki,Chugoku and Shikoku Districts  近畿・中国・四国地方': 'JMA',
	'Kyushu District  九州地方': 'JMA',
	'Ryukyu Is. District  琉球列島地方': 'JMA',
	'Geographical Survey Institute  国土地理院': 'GSI',
	# ---- ここから大学 → すべて "University" ----
	'Hokkaido University  北海道大学': 'University',
	'Hirosaki University  弘前大学': 'University',
	'Tohoku University  東北大学': 'University',
	'University of Tokyo  東京大学': 'University',
	'Nagoya University  名古屋大学': 'University',
	'Kyoto University  京都大学': 'University',
	'Kochi University  高知大学': 'University',
	'Kyushu University  九州大学': 'University',
	'Kagoshima University  鹿児島大学': 'University',
	# ---- 研究機関等 ----
	'National Research Institute for Earth Science and Disaster Prevention  国立研究開発法人防災科学技術研究所': 'NIED',
	'Japan Marine Science and Technology Center  国立研究開発法人海洋研究開発機構': 'JAMSTEC',
	'National Institute of Advanced Industrial Science and Technology  国立研究開発法人産業技術総合研究所': 'AIST',
	# ---- 都道府県・自治体 → Prefectural Gov. ----
	'Tokyo Metropolitan Government  東京都': 'Prefectural Gov.',
	'Aomori Prefecture  青森県': 'Prefectural Gov.',
	'Shizuoka Prefecture  静岡県': 'Prefectural Gov.',
	'Hot Springs Research Institute of Kanagawa Prefecture  神奈川県温泉地学研究所': 'Kanagawa HSRI',
	# ---- 観測網・その他 ----
	'Seismic Intensity Observation in Local Meteorological Observatory  気象官署の計測震度計': 'JMA Intensity',
	'F-net stations of National Research Institute for Earth Science and Disaster Prevention  国立研究開発法人防災科学技術研究所・広帯域地震観測網': 'NIED F-net',
	'Incorporated Research Institute for Seismology 米国大学間地震学研究連合(IRIS)': 'IRIS',
	'Association for the Development of Earthquake Prediction 公益財団法人地震予知総合研究振興会': 'ADEP',
	'Newly added stations (preliminary version) 新規追加観測点（暫定版リスト）': 'New (prelim.)',
}


_COMMENT_LABEL_MAP: dict[str, str] = {
	normalize_comment(k): v for k, v in _COMMENT_LABEL_MAP_RAW.items()
}


def _comment_to_affiliation_en(comment: str | float) -> str:
	key = normalize_comment(comment)
	label = _COMMENT_LABEL_MAP.get(key)
	if label is None:
		return 'Other'
	return label
