package com.ssafy.ssashinsa.heyfy.log;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.pattern.CompositeConverter;

import java.util.List;
import java.util.regex.Pattern;

public class MaskingConverter extends CompositeConverter<ILoggingEvent> {

    private static final Pattern EMAIL =
            Pattern.compile("([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+)");
    private static final Pattern RRN =
            Pattern.compile("([0-9]{6})-([0-9]{7})");
    private static final Pattern PHONE_SIMPLE =
            Pattern.compile("([0-9]{3})-?([0-9]{4})-?([0-9]{4})");
    private static final Pattern BEARER_AUTH =
            Pattern.compile("(?i)\\bAuthorization\\s*:\\s*Bearer\\s+[^\\s,]+");
    private static final Pattern TOKEN_KV =
            Pattern.compile("(?i)\\b(accessToken|refreshToken)\\s*[:=]\\s*([^\\s,;]+)");
    private static final Pattern ACCOUNT_NUMBER =
            Pattern.compile("([0-9]{2,3})-?([0-9]{5,6})-?([0-9]{1,4})");
    private static final Pattern CARD_NUMBER =
            Pattern.compile("([0-9]{4})-?([0-9]{4})-?([0-9]{4})-?([0-9]{4})");

    private record Rule(Pattern p, java.util.function.Function<java.util.regex.Matcher,String> fn) {}

    private final List<Rule> rules = List.of(
            new Rule(BEARER_AUTH, m -> "Authorization: Bearer ***"),
            new Rule(TOKEN_KV, m -> m.group(1) + "=***"),
            new Rule(RRN, m -> m.group(1) + "-*******"),
            new Rule(PHONE_SIMPLE, m -> m.group(1) + "-****-****"),
            new Rule(EMAIL, m -> "***@" + m.group(2)),
            new Rule(ACCOUNT_NUMBER, m -> m.group(1) + "-******-" + m.group(3)),
            new Rule(CARD_NUMBER, m -> m.group(1) + "-****-****-" + m.group(4))
    );

    @Override
    protected String transform(ILoggingEvent event, String in) {
        if (in == null || in.isEmpty()) return in;

        String out = in;
        for (Rule r : rules) {
            var matcher = r.p.matcher(out);
            StringBuffer sb = new StringBuffer();
            while (matcher.find()) {
                matcher.appendReplacement(sb,
                        java.util.regex.Matcher.quoteReplacement(r.fn.apply(matcher)));
            }
            matcher.appendTail(sb);
            out = sb.toString();
        }
        return out;
    }
}
