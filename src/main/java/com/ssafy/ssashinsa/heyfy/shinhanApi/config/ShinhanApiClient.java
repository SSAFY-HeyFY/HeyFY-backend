package com.ssafy.ssashinsa.heyfy.shinhanApi.config;

import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiProperties;
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
