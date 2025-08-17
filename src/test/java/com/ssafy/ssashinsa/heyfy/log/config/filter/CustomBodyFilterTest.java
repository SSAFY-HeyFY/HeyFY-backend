package com.ssafy.ssashinsa.heyfy.log;
import com.ssafy.ssashinsa.heyfy.log.config.CustomBodyFilter;
import org.junit.Test;
import org.zalando.logbook.BodyFilter;

import java.util.Set;

import static org.junit.Assert.*;

public class CustomBodyFilterTest {

    @Test
    public void filter_shouldMaskAllSensitiveFields() {
        Set<String> keysToRedact = Set.of(
                "email", "rrn", "accessToken", "refreshToken",
                "accountNumber", "cardNumber", "password"
        );
        BodyFilter filter = new CustomBodyFilter(keysToRedact);

        String json = """
            {
                "email": "user@domain.com",
                "rrn": "900101-1234567",
                "accessToken": "abc123",
                "refreshToken": "def456",
                "accountNumber": "123-456789-01",
                "cardNumber": "1234-5678-1234-5678",
                "password": "secret",
                "nested": {
                    "email": "nested@domain.com",
                    "password": "nestedSecret"
                }
            }
            """;

        String filtered = filter.filter("application/json", json);

        assertTrue(filtered.contains("\"email\":\"***\""));
        assertTrue(filtered.contains("\"rrn\":\"***\""));
        assertTrue(filtered.contains("\"accessToken\":\"***\""));
        assertTrue(filtered.contains("\"refreshToken\":\"***\""));
        assertTrue(filtered.contains("\"accountNumber\":\"***\""));
        assertTrue(filtered.contains("\"cardNumber\":\"***\""));
        assertTrue(filtered.contains("\"password\":\"***\""));
        assertTrue(filtered.contains("\"nested\":{\"email\":\"***\",\"password\":\"***\"}"));
    }
}