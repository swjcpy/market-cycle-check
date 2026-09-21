"""Plain-language text for the dashboard: what each cycle is, what hot/cold means, what a reader might consider.

Written for people with no finance background. Ideas follow Howard Marks' *Mastering the Market Cycle*, paraphrased:
cycles are driven by human emotion swinging between greed and fear; nobody can time the turns, but you can judge
where you are and lean cautious when things are hot and open-minded when they are cold. Nothing here is personal advice.
"""

# order shown on the page: the big emotional/financial cycles first, then the economy they sit on
ORDER = ["psychology", "credit", "economy", "policy", "profits", "realestate", "bonds", "distressed"]

CYCLE_INFO = {
    "psychology": dict(
        title="Investor mood", icon="🎭", up="more optimistic", down="more fearful",
        asks="Are investors relaxed and greedy, or scared?",
        measures="How calm the stock market's 'fear gauge' is, how far stock prices sit above their long-run trend, how expensive stocks are compared with a decade of company earnings, and how confident households feel.",
        hot="Investors are relaxed and optimistic. Prices have run up and few people are worried.",
        cold="Investors are frightened. Prices are depressed and pessimism is everywhere.",
        why="Prices are set by people, and people swing between too optimistic and too pessimistic. The mood matters more than the news.",
        hot_you="Marks argues risk is highest when everyone feels safe, though in our 30-year history some warm and hot stretches lasted for years. Consider whether you could sit through a 30% fall, and avoid chasing what is soaring.",
        cold_you="Marks argues bargains appear when everyone is scared, but prices can keep falling first. Patience and money you will not need soon matter most.",
        limit="Confidence surveys have been depressed by inflation worries since 2022, which pulls this gauge down even when markets are cheerful. Expensive stocks (a high CAPE) have historically told you more about returns over the next ten years than the next one.",
    ),
    "credit": dict(
        title="Lending & credit", icon="💳", up="loosening", down="tightening",
        asks="Is money easy or hard to borrow?",
        measures="How much extra interest companies must pay over the government, and whether banks are making loans easier or harder to get.",
        hot="Lenders are generous, even to shaky borrowers. Money is easy to borrow.",
        cold="Lenders have pulled back. Even good borrowers struggle to get loans.",
        why="Easy credit fuels booms, and the bad loans made during them cause the busts. Marks calls it the most important cycle.",
        hot_you="Cheap credit tempts everyone to borrow more. Be careful about taking on debt that only works if good times continue.",
        cold_you="Credit is scarce and lenders are careful. Tough for borrowers, but this is when careful lending earns the best returns.",
        limit="This measures investment-grade companies and bank surveys. It does not see the riskiest 'junk' lending, where excess shows up first.",
    ),
    "economy": dict(
        title="Jobs & the economy", icon="🏭", up="strengthening", down="weakening",
        asks="Is the economy growing or shrinking?",
        measures="New unemployment claims, whether the unemployment rate is rising, and whether factories are producing more than a year ago.",
        hot="The economy is growing strongly and jobs are plentiful.",
        cold="The economy is weak or shrinking, and unemployment is rising.",
        why="The economy moves in slow waves of expansion and recession. Good times do not last forever, and neither do bad ones.",
        hot_you="Strong times are a good moment to build savings and an emergency fund, because downturns always come back eventually.",
        cold_you="Downturns are painful, but recoveries follow every one. Job security and cash reserves matter most here.",
        limit="Government figures are revised after release. Factory output has grown more slowly for decades, so 'normal' growth reads slightly cool, and 2020-21 readings are distorted by the pandemic's low base.",
    ),
    "policy": dict(
        title="Central bank policy", icon="🏦", up="easing", down="tightening",
        asks="Is the central bank making money cheap or expensive?",
        measures="The central bank's interest rate after inflation, whether it has raised or cut rates over the past three months and the past year, and the gap between long-term and short-term rates.",
        hot="Money is cheap: interest rates are low and the central bank is helping the economy along.",
        cold="Money is expensive: rates are high or rising and the central bank is pushing on the brakes.",
        why="Governments and central banks try to smooth the cycle, which often changes its timing rather than removing it.",
        hot_you="Borrowing costs are low but savings earn little. Cheap money encourages risk-taking, which can build up problems.",
        cold_you="Savings earn more and borrowing costs more. Higher rates slow spending and often pressure markets for a while.",
        limit="The real rate subtracts recent inflation, so it looks very easy right after inflation spikes even if rates are rising.",
    ),
    "profits": dict(
        title="Company profits", icon="💰", up="rising", down="falling",
        asks="Are companies earning a lot or a little?",
        measures="How big a slice of the whole economy company profits take, and whether profits are growing.",
        hot="Companies are earning unusually large profits.",
        cold="Company profits are weak or falling.",
        why="Profits swing more than the economy itself. Unusually high profits attract competition, which tends to bring them back down.",
        hot_you="Record profits may already be reflected in prices. It is worth asking what happens if they simply return to normal.",
        cold_you="Weak profits may be priced in early. A recovery in earnings can arrive before it is obvious.",
        limit="Profits are at a record share of the economy (about 13%), and that share has risen for decades, so this gauge sits at 'hot' even when profits are not surging. Figures are revised.",
    ),
    "realestate": dict(
        title="Housing", icon="🏠", up="heating up", down="cooling",
        asks="Is the housing market booming or cooling?",
        measures="The extra cost of a mortgage over government bonds, home prices compared with rents, how fast prices are rising, and how many new homes are being permitted.",
        hot="Homes are expensive, prices are climbing fast, and builders are busy.",
        cold="Prices are falling or flat, building has slowed and mortgages cost more relative to government rates.",
        why="Property is bought with borrowed money, so booms and busts are large and slow.",
        hot_you="If you are buying, avoid stretching your budget on the assumption that prices keep rising. Do not count on further gains.",
        cold_you="When buyers are scarce and lenders cautious, prices can be reasonable, but only for people with stable income and a cash cushion.",
        limit="Housing data arrives about three months late and moves slowly.",
    ),
    "bonds": dict(
        title="Interest rates & bonds", icon="📈", up="bonds getting pricier", down="bonds getting cheaper",
        asks="Are bonds expensive or cheap?",
        measures="The extra yield investors demand for lending long-term, the 10-year interest rate versus its own average, and how much that rate changed over the year.",
        hot="Bond prices are high and yields are low. Lenders get little reward and investors are relaxed.",
        cold="Bonds are cheap and yields are high, often after a period of losses for bondholders.",
        why="Interest rates drive the price of nearly everything. Low rates push people into riskier things; high rates do the reverse.",
        hot_you="Low yields mean little income from safe lending, which tempts people to reach for riskier returns. Be aware of that pull.",
        cold_you="Higher yields mean safe savings and bonds pay more than in years, though recent buyers may be sitting on losses.",
        limit="Yields have been rising for several years, so this gauge has read 'cold' for a long stretch.",
    ),
    "distressed": dict(
        title="Distressed debt (rough proxy)", icon="🚨", up="distress easing", down="distress building",
        asks="Are many companies struggling to repay their debts?",
        measures="How many business loans banks are writing off (a slow, late signal), plus the same extra-interest gauge used in the lending cycle. So it partly repeats that gauge.",
        hot="Few companies are in trouble. Specialists who buy troubled debt have little to buy.",
        cold="Many companies are struggling to repay. Bargain hunters who specialise in bad debt are busy.",
        why="Marks describes distress as the cycle's lowest point: the moment when prospective returns are best, though the danger of further losses is still real.",
        hot_you="Calm on this gauge is normal in good times. It does not mean nothing can go wrong.",
        cold_you="This is where professional investors look for bargains. It is risky and expert territory, not a place for beginners.",
        limit="A ROUGH PROXY. True distressed-debt data (default rates, bond prices) is paid data we do not have. It overlaps with the lending gauge.",
    ),
}

# label, how to show the raw value, one-line plain meaning
INDICATOR_INFO = {
    "vix": ("Stock-market fear gauge (VIX)", "{:.1f}", "How much price swinging traders expect. Low = calm and complacent, high = scared."),
    "cape": ("Stock prices vs 10 years of earnings (CAPE)", "{:.1f}", "Stock prices divided by the average of the last ten years of company earnings, after inflation. Higher = more expensive. It has only been above 40 once before, in 1999-2000."),
    "sp500_vs_10y_trend": ("Stock prices vs 10-year trend", "{:+.0%}", "How far the S&P 500 sits above (+) or below (−) its average of the last 10 years."),
    "consumer_sentiment": ("Household confidence", "{:.0f}", "University of Michigan survey of how confident ordinary households feel."),
    "baa_10y_spread": ("Extra interest firms pay", "{:.2f} points", "Extra yearly interest that fairly safe companies pay over the U.S. government. Small = lenders relaxed."),
    "sloos_ci_tightening": ("Banks tightening loans", "{:+.0f}%", "Net share of banks making business loans harder to get. Negative = banks are easing."),
    "real_policy_rate": ("Interest rate after inflation", "{:+.1f} points", "The central bank's rate minus recent inflation: how tight money really is."),
    "policy_rate_3m_change": ("Rate change over 3 months", "{:+.2f} points", "How much the central bank's rate rose (+) or fell (−) in the past three months. Shows a fresh hike or cut quickly."),
    "policy_rate_12m_change": ("Rate change over 12 months", "{:+.2f} points", "How much the central bank's rate rose (+) or fell (−) in the past year."),
    "curve_10y_minus_3m": ("Long minus short rates", "{:+.2f} points", "10-year rate minus 3-month rate. Below zero ('inverted') has often come before recessions."),
    "jobless_claims_yoy": ("New jobless claims vs last year", "{:+.0f}%", "New unemployment claims compared with a year ago (4-week average)."),
    "unemployment_12m_change": ("Unemployment change over 12 months", "{:+.1f} points", "How much the unemployment rate rose (+) or fell (−) over the year."),
    "industrial_output_yoy": ("Factory output vs last year", "{:+.1f}%", "Output of factories, mines and utilities compared with a year ago."),
    "profit_share_of_gdp": ("Profits' share of the economy", "{:.1%}", "After-tax company profits as a share of everything the country produces."),
    "profit_growth_yoy": ("Profit growth vs last year", "{:+.0f}%", "After-tax company profits compared with a year earlier."),
    "mortgage_spread": ("Mortgage rate over bonds", "{:.2f} points", "How much more a mortgage costs than a 10-year government bond. Wide = lenders nervous."),
    "price_to_rent": ("Home prices vs rents", "{:.2f}", "Home price index divided by the rent index. Higher = buying costs more relative to renting."),
    "house_price_yoy": ("Home price growth", "{:+.1f}%", "Change in home prices over the past year."),
    "building_permits_yoy": ("New-home permits vs last year", "{:+.0f}%", "Permits to build new homes compared with a year ago."),
    "term_premium": ("Extra yield for long bonds", "{:.2f} points", "Extra yield investors demand for holding 10-year bonds instead of short ones."),
    "yield_vs_10y_average": ("10-year rate vs its average", "{:+.2f} points", "Today's 10-year interest rate minus its own 10-year average."),
    "yield_12m_change": ("10-year rate change over 12 months", "{:+.2f} points", "How much the 10-year interest rate rose (+) or fell (−) in the past year."),
    "business_loan_chargeoffs": ("Business loans written off", "{:.2f}%", "Share of business loans banks gave up on during the year."),
    "credit_spread_level": ("Extra interest firms pay (level)", "{:.2f} points", "Same signal as in the lending gauge: how much extra interest companies pay."),
}

# indicators whose stored value is a fraction that the format above turns into a percent, or a raw number
BANDS = [  # (unused threshold: summary.py owns the edges, symmetric at 0.35 / 1.0), key, plain word, colour role
    (1.0, "hot", "Hot", "hot"),
    (0.35, "warm", "Warm", "warm"),
    (-0.35, "normal", "Normal", "normal"),
    (-1.0, "cool", "Cool", "cool"),
    (-9.0, "cold", "Cold", "cold"),
]

# Action lists are deliberately the SAME for every band: risk hygiene that holds in any market. Our own 30-year history
# did not show that hot stages were followed by more big falls, so we do not tell readers to act differently when hot.
COMMON_DO = ["Keep an emergency fund in cash, enough for several months of expenses.",
             "Only invest money you will not need for many years, and spread purchases over time.",
             "Check that what you own matches how large a fall you could truly live with."]
COMMON_AVOID = ["Do not borrow money to invest.",
                "Do not make big changes out of fear or excitement; the gauges below cannot tell you when a turn will come."]

HEADLINE = {
    "hot": dict(
        title="Overall: running hot",
        summary="Lending and investor mood together are running hot. Howard Marks, the author behind these ideas, argues that risk quietly builds when everyone feels safe. In our own 30-year history, hot periods were not followed by more big falls, and some hot-or-warm stretches lasted for years, so treat this as a reminder to check your risk, not as a signal to sell."),
    "warm": dict(
        title="Overall: leaning optimistic",
        summary="Taken together, conditions lean cheerful, but not at an extreme. The two gauges behind this headline (lending and investor mood) can disagree, so the line below shows which one is driving it."),
    "normal": dict(
        title="Overall: normal, no strong signal",
        summary="Lending and investor mood are in their usual range. The gauges are not giving a strong hint either way."),
    "cool": dict(
        title="Overall: leaning cautious",
        summary="Taken together, conditions lean cautious, but not at an extreme. The two gauges behind this headline (lending and investor mood) can disagree, so the line below shows which one is driving it."),
    "cold": dict(
        title="Overall: running cold, fear is high",
        summary="Lending and investor mood together are running cold. Howard Marks argues the best opportunities appear when others are afraid, but prices can keep falling first. In our history, the biggest further falls came when fear was spreading."),
}
for _h in HEADLINE.values():
    _h["do"], _h["avoid"] = COMMON_DO, COMMON_AVOID

STAGE = {  # (band group, direction) -> (short label, sentence)
    ("hot", "up"): ("Heating up", "Optimism is still building."),
    ("hot", "steady"): ("Running hot", "Conditions are hot and not changing much."),
    ("hot", "down"): ("Cooling from a high", "Still hot, but the mood is starting to turn."),
    ("normal", "up"): ("Warming", "Conditions are normal but improving."),
    ("normal", "steady"): ("Steady", "Conditions are in their normal range."),
    ("normal", "down"): ("Cooling", "Conditions are normal but softening."),
    ("cold", "up"): ("Recovering", "Cold but improving."),
    ("cold", "steady"): ("Cold", "Fear is high and not changing much."),
    ("cold", "down"): ("Getting colder", "Conditions are worsening. In our 30-year history the biggest further falls came in this stage, from two episodes: the 2000-02 and 2007-09 bear markets."),
}

# softer wording when the gauge is only warm or cool (not hot or cold)
STAGE_MILD_TEXT = {
    ("hot", "up"): "Getting more cheerful.", ("hot", "steady"): "Somewhat cheerful, not changing much.", ("hot", "down"): "Somewhat cheerful, but cooling.",
    ("cold", "up"): "Somewhat cautious, but improving.", ("cold", "steady"): "Somewhat cautious, not changing much.", ("cold", "down"): "Getting more cautious.",
}

# A reading counted at less than full strength must say why (shown next to it on the page)
CONFIDENCE_NOTE = {
    "consumer_sentiment": "Given half the weight it would otherwise get: this survey measures households' worries about prices more than investors' appetite for risk, "
                          "it changed from phone to online interviews in 2024 (so recent readings are not comparable with older ones), and it is near its lowest level on record.",
    "credit_spread_level": "Given half the weight it would otherwise get: this is the same series as the lending gauge's main reading, so counted in full "
                           "this gauge would mostly repeat that one instead of giving a separate view of distress.",
    "profit_share_of_gdp": "Given half the weight it would otherwise get: company profits have taken a much bigger share of the economy since about 2005, so this "
                           "reading sits near the top of its range most of the time and says little about where we are in the cycle.",
}

NOT_A_SIGNAL = "This is general education, not a buy or sell signal and not personal advice."

DISCLAIMER = ("This page is general education, not personal financial advice. It measures the mood and conditions of the "
              "market cycle; it cannot predict the future. Over the past 30 years these gauges reflected the known booms and busts "
              "but did not reliably predict stock returns. Talk to a qualified adviser before making decisions about your money.")


# Standard "leading / coincident / lagging" behaviour of each reading in the ECONOMY, from the Conference Board's business-cycle
# indexes where the reading (or its close relative) is a component ("index"), otherwise our judgement ("judgement") or not classified.
# This is textbook knowledge, NOT something measured here; the measured lead/lag against the stock market is separate.
INDICATOR_TIMING = {
    "vix": ("none", "judgement"), "sp500_vs_10y_trend": ("leading", "index"), "cape": ("longrun", "judgement"),
    "consumer_sentiment": ("leading", "convention"),
    "baa_10y_spread": ("leading", "convention"), "sloos_ci_tightening": ("leading", "judgement"),
    "real_policy_rate": ("leading", "judgement"), "policy_rate_12m_change": ("leading", "judgement"),
    "policy_rate_3m_change": ("leading", "judgement"), "curve_10y_minus_3m": ("leading", "index"),
    "jobless_claims_yoy": ("leading", "index"), "unemployment_12m_change": ("lagging", "convention"),
    "industrial_output_yoy": ("coincident", "index"),
    "profit_share_of_gdp": ("coincident", "judgement"), "profit_growth_yoy": ("coincident", "judgement"),
    "mortgage_spread": ("none", "judgement"), "price_to_rent": ("lagging", "judgement"), "house_price_yoy": ("lagging", "judgement"),
    "building_permits_yoy": ("leading", "index"),
    "term_premium": ("none", "judgement"), "yield_vs_10y_average": ("none", "judgement"), "yield_12m_change": ("none", "judgement"),
    "business_loan_chargeoffs": ("lagging", "judgement"), "credit_spread_level": ("leading", "convention"),
}
TIMING_TEXT = {
    "leading": "Usually turns before the economy does",
    "coincident": "Usually moves with the economy",
    "lagging": "Usually turns after the economy does",
    "longrun": "Says more about returns over years than about timing",
    "none": "Not classified as leading or lagging",
}
TIMING_BASIS = {"index": "a Conference Board index component or close relative", "convention": "a widely used classification", "judgement": "our judgement, not measured here"}

LEADLAG_INTRO = "Measured on our own history against the S&P 500 stock index, so treat every number as rough: only a handful of big market falls happened in that time."
