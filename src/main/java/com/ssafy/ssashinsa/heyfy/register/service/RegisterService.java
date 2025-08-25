package com.ssafy.ssashinsa.heyfy.register.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.ssashinsa.heyfy.account.domain.Account;
import com.ssafy.ssashinsa.heyfy.account.domain.ForeignAccount;
import com.ssafy.ssashinsa.heyfy.account.repository.AccountRepository;
import com.ssafy.ssashinsa.heyfy.account.repository.ForeignAccountRepository;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.common.util.SecurityUtil;
import com.ssafy.ssashinsa.heyfy.inquire.exception.ShinhanInquireApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.create.ShinhanCreateDepositRequestDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.create.ShinhanCreateDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireSingleDepositRequestDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireSingleDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.foreign.ShinhanCreateforeignDepositRequestDto;
import com.ssafy.ssashinsa.heyfy.register.exception.ShinhanRegisterApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.common.ShinhanCommonRequestHeaderDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import reactor.core.publisher.Mono;

@Service
@Slf4j
@RequiredArgsConstructor
public class RegisterService {

    private final UserRepository userRepository;
    private final AccountRepository accountRepository;
    private final ForeignAccountRepository foreignAccountRepository;


    private final ShinhanApiClient shinhanApiClient;
    private final ShinhanApiUtil shinhanApiUtil;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public ShinhanCreateDepositResponseDto createDepositAccount() {
        try {
            String apiKey = shinhanApiClient.getManagerKey();
            String accountTypeUniqueNo = shinhanApiClient.getAccountTypeUniqueNo();

            String studentId = SecurityUtil.getCurrentStudentId();
            Users user = userRepository.findByStudentId(studentId)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

            String userKey = user.getUserKey();
            if (userKey == null || userKey.isEmpty()) {
                throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
            }


            if (accountRepository.findByUser(user).isPresent()) {
                log.warn("이미 계좌가 존재하는 유저입니다. studentId: {}", studentId);
                throw new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_ALREADY_EXISTS);
            }

            ShinhanCommonRequestHeaderDto commonHeaderDto = shinhanApiUtil.createHeaderDto(
                    "createDemandDepositAccount",
                    "createDemandDepositAccount",
                    apiKey,
                    userKey
            );


            ShinhanCreateDepositRequestDto requestDto = ShinhanCreateDepositRequestDto.builder()
                    .Header(commonHeaderDto)
                    .accountTypeUniqueNo(accountTypeUniqueNo)
                    .build();

            logRequest(requestDto);

            ShinhanCreateDepositResponseDto response = shinhanApiClient.getClient("edu")
                    .post()
                    .uri("/demandDeposit/createDemandDepositAccount")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, r ->
                            r.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ShinhanRegisterApiErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(ShinhanCreateDepositResponseDto.class)
                    .doOnNext(this::logResponse)
                    .block();

            String accountNo = response.getREC().getAccountNo();


            Account account = Account.builder()
                    .user(user)
                    .accountNo(accountNo)
                    .build();
            accountRepository.save(account);


            return response;

        } catch (CustomException ce) {
            log.error("커스텀 예외 발생: {}", ce.getMessage());
            throw ce;
        } catch (Exception e) {
            log.error("계좌 개설 API 호출 실패 : {}", e.getMessage(), e);
            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
        }
    }

    public ShinhanCreateDepositResponseDto createForeignDepositAccount() {
        try {
            String apiKey = shinhanApiClient.getManagerKey();
            String accountTypeUniqueNo = shinhanApiClient.getForeignAccountTypeUniqueNo();

            String studentId = SecurityUtil.getCurrentStudentId();
            Users user = userRepository.findByStudentId(studentId)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

            String userKey = user.getUserKey();
            if (userKey == null || userKey.isEmpty()) {
                throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
            }

            ShinhanCommonRequestHeaderDto commonHeaderDto = shinhanApiUtil.createHeaderDto(
                    "createForeignCurrencyDemandDepositAccount",
                    "createForeignCurrencyDemandDepositAccount",
                    apiKey,
                    userKey
            );

            ShinhanCreateforeignDepositRequestDto requestDto = ShinhanCreateforeignDepositRequestDto.builder()
                    .Header(commonHeaderDto)
                    .accountTypeUniqueNo(accountTypeUniqueNo)
                    .currency("USD") // 예시로 USD를 사용, 필요에 따라 변경
                    .build();


            if (foreignAccountRepository.findByUser(user).isPresent()) {
                log.warn("이미 계좌가 존재하는 유저입니다. studentId: {}", studentId);

                throw new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_ALREADY_EXISTS);
            }

            logRequest(requestDto);

            ShinhanCreateDepositResponseDto response = shinhanApiClient.getClient("edu")
                    .post()
                    .uri("/demandDeposit/foreignCurrency/createForeignCurrencyDemandDepositAccount")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, r ->
                            r.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ShinhanRegisterApiErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(ShinhanCreateDepositResponseDto.class)
                    .doOnNext(this::logResponse)
                    .block();

            String accountNo = response.getREC().getAccountNo();
            String currencyCode = response.getREC().getCurrency().getCurrency();


            ForeignAccount foreignAccount = ForeignAccount.builder()
                    .user(user)
                    .accountNo(accountNo)
                    .currency(currencyCode)
                    .build();
            foreignAccountRepository.save(foreignAccount);

            return response;

        } catch (CustomException ce) {
            log.error("커스텀 예외 발생: {}", ce.getMessage());
            throw ce;
        } catch (Exception e) {
            log.error("계좌 개설 API 호출 실패 : {}", e.getMessage(), e);
            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
        }
    }


    private void logRequest(Object requestDto) {
        try {
            log.info("Request JSON: {}", objectMapper.writeValueAsString(requestDto));
        } catch (Exception e) {
            log.error("Request logging error", e);
        }
    }

    private void logResponse(Object responseDto) {
        try {
            log.info("Response JSON: {}", objectMapper.writeValueAsString(responseDto));
        } catch (Exception e) {
            log.error("Response logging error", e);
        }
    }


    @Transactional
    public String registerAccountFromShinhan(String accountNo) {
        String studentId = SecurityUtil.getCurrentStudentId();
        log.info("학생 ID [{}]의 계좌 등록을 시작합니다.", studentId);

        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        String apiKey = shinhanApiClient.getManagerKey();
        String userKey = user.getUserKey();

        ShinhanCommonRequestHeaderDto commonHeaderDto = shinhanApiUtil.createHeaderDto(
                "inquireDemandDepositAccount",
                "inquireDemandDepositAccount",
                apiKey,
                userKey
        );

        ShinhanInquireSingleDepositRequestDto requestDto = ShinhanInquireSingleDepositRequestDto.builder()
                .Header(commonHeaderDto)
                .accountNo(accountNo)
                .build();

        try {
            ShinhanInquireSingleDepositResponseDto response = shinhanApiClient.getClient("edu")
                    .post()
                    .uri("/demandDeposit/inquireDemandDepositAccount")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, r ->
                            r.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ShinhanInquireApiErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(ShinhanInquireSingleDepositResponseDto.class)
                    .doOnNext(this::logResponse)
                    .block();

            registerAccount(accountNo);

        }  catch (Exception e) {
            log.error("계좌 등록 API 호출 실패 : {}", e.getMessage(), e);
            throw e;
        }

        return accountNo;
    }

    @Transactional
    public String registerForeignAccountFromShinhan(String accountNo) {
        String studentId = SecurityUtil.getCurrentStudentId();
        log.info("학생 ID [{}]의 외화 계좌 등록을 시작합니다.", studentId);

        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        String apiKey = shinhanApiClient.getManagerKey();
        String userKey = user.getUserKey();

        ShinhanCommonRequestHeaderDto commonHeaderDto = shinhanApiUtil.createHeaderDto(
                "inquireForeignCurrencyDemandDepositAccount",
                "inquireForeignCurrencyDemandDepositAccount",
                apiKey,
                userKey
        );

        ShinhanInquireSingleDepositRequestDto requestDto = ShinhanInquireSingleDepositRequestDto.builder()
                .Header(commonHeaderDto)
                .accountNo(accountNo)
                .build();

        try {
            ShinhanInquireSingleDepositResponseDto response = shinhanApiClient.getClient("edu")
                    .post()
                    .uri("/demandDeposit/foreignCurrency/inquireForeignCurrencyDemandDepositAccount")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, r ->
                            r.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ShinhanInquireApiErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(ShinhanInquireSingleDepositResponseDto.class)
                    .doOnNext(this::logResponse)
                    .block();

            registerForeignAccount(accountNo, response.getREC().getCurrency());

        }  catch (Exception e) {
            log.error("계좌 등록 API 호출 실패 : {}", e.getMessage(), e);
            throw e;
        }

        return accountNo;
    }




    @Transactional
    public void registerAccount(String accountNo) {
        String studentId = SecurityUtil.getCurrentStudentId();
        log.info("학생 ID [{}]의 계좌 등록을 시작합니다.", studentId);


            Users user = userRepository.findByStudentId(studentId)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        try {
            if (accountRepository.existsByUser(user)) {
                user.setAccount(null);

                accountRepository.deleteByUser(user);

                user = userRepository.getReferenceById(user.getId());
            }

            // 3) 새 계좌 insert
            Account newAccount = Account.builder()
                    .user(user)
                    .accountNo(accountNo)
                    .build();

            accountRepository.save(newAccount);


            log.info("학생 ID [{}]의 일반 계좌가 성공적으로 개설되었습니다. 계좌번호: {}", user.getStudentId(), accountNo);

        } catch (CustomException ce) {
            log.error("학생 ID [{}]의 일반 계좌 개설 중 커스텀 예외 발생: {}", user.getStudentId(), ce.getMessage());
        } catch (Exception e) {
            log.error("학생 ID [{}]의 일반 계좌 개설 API 호출 실패: {}", user.getStudentId(), e.getMessage(), e);
        }
    }

    @Transactional
    public void registerForeignAccount(String accountNo, String currency) {
        String studentId = SecurityUtil.getCurrentStudentId();
        log.info("학생 ID [{}]의 외화 계좌 등록을 시작합니다.", studentId);


        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        try {
            if (foreignAccountRepository.existsByUser(user)) {
                user.setAccount(null);

                foreignAccountRepository.deleteByUser(user);

                user = userRepository.getReferenceById(user.getId());
            }

            ForeignAccount newAccount = ForeignAccount.builder()
                    .user(user)
                    .accountNo(accountNo)
                    .currency(currency)
                    .build();

            foreignAccountRepository.save(newAccount);


            log.info("학생 ID [{}]의 외환 계좌가 성공적으로 개설되었습니다. 계좌번호: {}", user.getStudentId(), accountNo);

        } catch (CustomException ce) {
            log.error("학생 ID [{}]의 외환 계좌 개설 중 커스텀 예외 발생: {}", user.getStudentId(), ce.getMessage());
        } catch (Exception e) {
            log.error("학생 ID [{}]의 외환 계좌 개설 API 호출 실패: {}", user.getStudentId(), e.getMessage(), e);
        }
    }

    // 외부에서 직접 호출되지 않도록 private 메서드로 변경
    private void createForeignDepositAccount(Users user) {
        try {
            String apiKey = shinhanApiClient.getManagerKey();
            String accountTypeUniqueNo = shinhanApiClient.getForeignAccountTypeUniqueNo();

            String userKey = user.getUserKey();
            if (userKey == null || userKey.isEmpty()) {
                throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
            }

            ShinhanCommonRequestHeaderDto commonHeaderDto = shinhanApiUtil.createHeaderDto(
                    "createForeignCurrencyDemandDepositAccount",
                    "createForeignCurrencyDemandDepositAccount",
                    apiKey,
                    userKey
            );

            ShinhanCreateforeignDepositRequestDto requestDto = ShinhanCreateforeignDepositRequestDto.builder()
                    .Header(commonHeaderDto)
                    .accountTypeUniqueNo(accountTypeUniqueNo)
                    .currency("USD")
                    .build();

            logRequest(requestDto);

            ShinhanCreateDepositResponseDto response = shinhanApiClient.getClient("edu")
                    .post()
                    .uri("/demandDeposit/foreignCurrency/createForeignCurrencyDemandDepositAccount")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, r ->
                            r.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ShinhanRegisterApiErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(ShinhanCreateDepositResponseDto.class)
                    .doOnNext(this::logResponse)
                    .block();

            String accountNo = response.getREC().getAccountNo();
            String currencyCode = response.getREC().getCurrency().getCurrency();

            ForeignAccount foreignAccount = ForeignAccount.builder()
                    .user(user)
                    .accountNo(accountNo)
                    .currency(currencyCode)
                    .build();
            foreignAccountRepository.save(foreignAccount);

            log.info("학생 ID [{}]의 외화 계좌가 성공적으로 개설되었습니다. 계좌번호: {}", user.getStudentId(), accountNo);

        } catch (CustomException ce) {
            log.error("학생 ID [{}]의 외화 계좌 개설 중 커스텀 예외 발생: {}", user.getStudentId(), ce.getMessage());
        } catch (Exception e) {
            log.error("학생 ID [{}]의 외화 계좌 개설 API 호출 실패: {}", user.getStudentId(), e.getMessage(), e);
        }
    }


}