package com.ssafy.ssashinsa.heyfy.transfer.external;

import com.ssafy.ssashinsa.heyfy.transfer.config.SsafyFinApiProperties;
import com.ssafy.ssashinsa.heyfy.transfer.exception.CustomExceptions;
import com.ssafy.ssashinsa.heyfy.transfer.dto.TransferRequestBody;
import com.ssafy.ssashinsa.heyfy.transfer.dto.TransferResponseBody;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

@Slf4j
@Service
@RequiredArgsConstructor
public class SsafyFinApiClient {

    private final SsafyFinApiProperties properties;
    private RestClient restClient;

    @PostConstruct
    public void init() {
        this.restClient = RestClient.builder()
                .baseUrl(properties.baseUrl())
                .defaultHeader("Content-Type", MediaType.APPLICATION_JSON_VALUE)
                .build();
    }

    public TransferResponseBody transfer(TransferRequestBody requestBody) {
        try {
            log.info("외부 금융 API 이체 요청 실행");
            return restClient.post()
                    .uri("/ssafy/api/v1/edu/demandDeposit/updateDemandDepositAccountTransfer")
                    .body(requestBody)
                    .retrieve()
                    .body(TransferResponseBody.class);
        } catch (RestClientException e) {
            log.error("SSAFY FinAPI 호출 실패: {}", e.getMessage());
            throw new CustomExceptions.ExternalApiCallException("외부 금융 API 호출에 실패했습니다. 잠시 후 다시 시도해주세요.");
        }
    }
}