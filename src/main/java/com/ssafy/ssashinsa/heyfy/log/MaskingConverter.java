package com.ssafy.ssashinsa.heyfy.log;

import ch.qos.logback.classic.pattern.ClassicConverter;
import ch.qos.logback.classic.spi.ILoggingEvent;

import java.util.List;
import java.util.regex.Pattern;

public class MaskingConverter extends ClassicConverter {

    private static final Pattern EMAIL =
            Pattern.compile("([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+)");
    private static final Pattern RRN =
            Pattern.compile("([0-9]{6})-?([1-4][0-9]{6})");
    private static final Pattern PHONE_SIMPLE =
            Pattern.compile("([0-9]{3})-?([0-9]{4})-?([0-9]{4})");
    private static final Pattern BEARER_AUTH =
            Pattern.compile("(?i)\\bauthorization\\s*:\\s*bearer\\s+[^\\s,]+");
    private static final Pattern TOKEN_KV =
            Pattern.compile("(?i)\\b(access[_-]?token|refresh[_-]?token|authorization)\\s*[:=]\\s*([^\\s,;]+)");
    private static final Pattern ACCOUNT_NUMBER =
            Pattern.compile("([0-9]{2,4})-?([0-9]{2,6})-?([0-9]{1,4})");
    private static final Pattern CARD_NUMBER =
            Pattern.compile("([0-9]{4})-?([0-9]{4})-?([0-9]{4})-?([0-9]{4})");

    private record Rule(Pattern p, java.util.function.Function<java.util.regex.Matcher,String> fn) {}

    private final List<Rule> rules = List.of(
            // Authorization: Bearer ***
            new Rule(BEARER_AUTH, m -> "Authorization: Bearer ***"),
            // key=value → key=***
            new Rule(TOKEN_KV, m -> m.group(1) + "=***"),
            // 주민등록번호
            new Rule(RRN, m -> m.group(1) + "-*******"),
            // 휴대전화
            new Rule(PHONE_SIMPLE, m -> m.group(1) + "-****-****"),
            // 이메일
            new Rule(EMAIL, m -> "***@" + m.group(2)),
            // 계좌번호
            new Rule(ACCOUNT_NUMBER, m -> m.group(1) + "-******-" + m.group(3)),
            // 카드번호
            new Rule(CARD_NUMBER, m -> m.group(1) + "-****-****-" + m.group(4))
    );

    @Override
    public String convert(ILoggingEvent event) {
        String msg = event.getFormattedMessage();
        if (msg == null || msg.isEmpty()) return msg;

        String out = msg;
        for (Rule r : rules) {
            var matcher = r.p.matcher(out);
            StringBuffer sb = new StringBuffer();
            while (matcher.find()) {
                matcher.appendReplacement(sb, java.util.regex.Matcher.quoteReplacement(r.fn.apply(matcher)));
            }
            matcher.appendTail(sb);
            out = sb.toString();
        }
        return out;
    }
}