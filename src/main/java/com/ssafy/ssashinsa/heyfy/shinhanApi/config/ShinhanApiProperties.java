package com.ssafy.ssashinsa.heyfy.shinhanApi.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;
import java.util.HashMap;
import java.util.Map;


@Configuration
@ConfigurationProperties(prefix = "external.apis")
@Getter @Setter
public class ShinhanApiProperties {
    private Common common;
    private Map<String, Domain> domains = new HashMap<>();

    @Getter @Setter
    public static class Common {
        private String managerKey;
        private String accountTypeUniqueNo;
        private String foreignAccountTypeUniqueNo;
    }

    @Getter @Setter
    public static class Domain {
        private String baseUrl;
    }
}
