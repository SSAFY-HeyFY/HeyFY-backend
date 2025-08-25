package com.ssafy.ssashinsa.heyfy.fastapi.client;

import com.ssafy.ssashinsa.heyfy.fastapi.config.FastApiProperties;
import com.ssafy.ssashinsa.heyfy.fastapi.dto.FastApiRealTimeRatesDto;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

@Slf4j
@Component
@RequiredArgsConstructor
public class FastApiClient {
    private final FastApiProperties fastApiProperties;
    private final WebClient.Builder webClientBuilder;

    public FastApiRealTimeRatesDto getRealTimeRates(){
        FastApiRealTimeRatesDto response = getClient()
                .get()
                .uri("/realtime-rates")
                .header("Content-Type", "application/json")
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            throw new IllegalStateException("Failed to fetch real-time rates from FastAPI.");
                        }))
                .bodyToMono(FastApiRealTimeRatesDto.class)
                .block();
        if(response==null){
            throw new IllegalStateException("Failed to fetch real-time rates from FastAPI.");
        }
        return response;
    }
    private WebClient getClient() {
        String baseUrl = fastApiProperties.getFullBaseUrl() + "/api";

        if(baseUrl==null){
            throw new IllegalArgumentException("FastAPI baseUrl is not configured.");
        }
        if (!(baseUrl.startsWith("http://") || baseUrl.startsWith("https://"))) {
            throw new IllegalArgumentException("FastAPI baseUrl must start with http:// or https://. Current value: " + baseUrl);
        }

        return webClientBuilder
                .baseUrl(baseUrl)
                .build();
    }
}
