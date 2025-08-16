package com.ssafy.ssashinsa.heyfy.log;

import ch.qos.logback.classic.pattern.ClassicConverter;
import ch.qos.logback.classic.spi.ILoggingEvent;

public class MaskingConverter extends ClassicConverter {

    @Override
    public String convert(ILoggingEvent event) {
        return "";
    }
}