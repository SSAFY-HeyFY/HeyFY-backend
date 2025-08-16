package com.ssafy.ssashinsa.heyfy.log;

import ch.qos.logback.classic.spi.ILoggingEvent;
import org.junit.Test;
import org.mockito.Mockito;

import static org.junit.Assert.*;

public class MaskingConverterTest {

    private final MaskingConverter converter = new MaskingConverter();

    private String convert(String msg) {
        ILoggingEvent event = Mockito.mock(ILoggingEvent.class);
        Mockito.when(event.getFormattedMessage()).thenReturn(msg);
        return converter.convert(event);
    }

    @Test
    public void testEmailMasking() {
        assertEquals("***@domain.com", convert("user@domain.com"));
    }

    @Test
    public void testRrnMasking() {
        assertEquals("900101-*******", convert("900101-1234567"));
    }

    @Test
    public void testPhoneMasking() {
        assertEquals("010-****-****", convert("010-1234-5678"));
    }

    @Test
    public void testBearerAuthMasking() {
        assertEquals("Authorization: Bearer ***", convert("Authorization: Bearer abcdefg12345"));
    }

    @Test
    public void testTokenKvMasking() {
        assertEquals("accessToken=***", convert("accessToken=abcdefg12345"));
        assertEquals("refreshToken=***", convert("refreshToken:xyz"));
    }

    @Test
    public void testAccountNumberMasking() {
        assertEquals("123-******-01", convert("123-456789-01"));
    }

    @Test
    public void testCardNumberMasking() {
        assertEquals("1234-****-****-5678", convert("1234-5678-1234-5678"));
    }

    @Test
    public void testMultipleMasking() {
        String input = "user@domain.com 010-1234-5678 900101-1234567 123-456789-01 1234-5678-1234-5678";
        String expected = "***@domain.com 010-****-**** 900101-******* 123-******-01 1234-****-****-5678";
        assertEquals(expected, convert(input));
    }
}