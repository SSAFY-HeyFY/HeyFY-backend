package com.ssafy.ssashinsa.heyfy.shinhanApi.config;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

@Component
@RequiredArgsConstructor
public class ShinhanApiClient {

    private final ShinhanApiProperties apiProperties;
    private final WebClient.Builder webClientBuilder;

    public String getManagerKey() {
        return apiProperties.getCommon().getManagerKey();
    }
    public String getAccountTypeUniqueNo() {
        return apiProperties.getCommon().getAccountTypeUniqueNo();
    }
    public String getForeignAccountTypeUniqueNo() {
        return apiProperties.getCommon().getForeignAccountTypeUniqueNo();
    }
    public WebClient getClient(String domain) {
        ShinhanApiProperties.Domain domainConfig = apiProperties.getDomains().get(domain);

        if (domainConfig == null) {
            throw new IllegalArgumentException("Unknown API domain: " + domain);
        }

        return webClientBuilder
                .baseUrl(domainConfig.getBaseUrl())
//                .defaultHeader("apiKey", apiProperties.getCommon().getManagerKey())
                .build();
    }
}
